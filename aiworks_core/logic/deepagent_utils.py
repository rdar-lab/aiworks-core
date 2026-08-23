"""
Deep agent utilities for running multi-phase agentic workflows.

Provides:
- run_deep_agent: Execute a deep agent with tools, VFS seeding, and output file collection
- recover_deep_agent_run: Attempt to complete an interrupted deep agent run
- save_generation_output_files: Persist agent output files as AttachedFile records
- generate_kb_manifest: Build KB manifest from attached KnowledgeBases
- prepare_attached_files_for_session: Build VFS seed files from KB and input files
- preprocess_kb_files: Load and summarize all KB files for a session
- prepare_tools_for_session: Build LangChain tools for a session (KB RAG, MCP, web)
"""
import base64
import datetime
import logging
from typing import Any, Dict, List, Optional, Union
import os
from pathlib import Path

from .session_helper import parse_session_to_file
from ..utils import async_to_sync
from django.conf import settings
from .deepagent_validation_tools import get_deepagent_validation_tools
from .document_parser import can_extract_text
from .files import get_file_summary, get_text_content, set_binary_content
from .llm import AgentCallbacks, get_playwright_tools, get_web_tools, invoke_llm, spawn_subagent, \
    CustomLLMResponseValidator
from .logic_utils import is_text_file, is_base64
from .mcp_tools import build_mcp_tools_from_servers
from .mcp_tunnel import build_tunnel_tools
from .rag import build_kb_rag_tools
from .schema_validation_utils import validate_json_with_schema_file
from ..models import AttachedFile, LLMConfiguration, Session, Tunnel

logger = logging.getLogger(__name__)

_RECOVERY_MAX_ATTEMPTS = 3


def get_files_summary(session: Session) -> str:
    # noinspection PyUnresolvedReferences
    files = session.attached_files.filter(
        file_type=AttachedFile.FILE_TYPE_INPUT
    )
    if not files:
        return ""

    lines = []
    for f in files:
        summary = get_file_summary(f)
        if summary:
            lines.append(f"FILE: {f.name}\nSUMMARY: {summary}")
        else:
            lines.append(f"FILE: {f.name}")

    return "\n\n".join(lines)


def preprocess_kb_files(session: Session, update_task_callback=None):
    """Load and summarise KB files attached to the session.

    Returns (kb_files, error) where kb_files is a list of {name, content} dicts.
    """
    kb_files = []
    if session.knowledge_bases.exists():
        unauthorized = session.knowledge_bases.exclude(user=session.user)
        if unauthorized.exists():
            raise PermissionError(
                f"Session {session.id} has knowledge bases belonging to another user: "
                f"{list(unauthorized.values_list('id', flat=True))}"
            )
        if update_task_callback:
            update_task_callback(progress_step="Indexing Knowledge Base ...")
        attached_kbs = list(session.knowledge_bases.prefetch_related("kb_files").all())
        for kb in attached_kbs:
            for f in kb.kb_files.all():
                # Make sure the file summarized
                get_file_summary(f)
                text = get_text_content(f)
                kb_files.append({"name": f.name, "content": text})
    return kb_files, None

def mount_schema_to_attached_files(schema_file_path, attached_files):
    if schema_file_path:
        schema_file_simple_path = '/' + schema_file_path.split("/")[-1]
        if schema_file_simple_path not in attached_files:
            try:
                schema_full_file_path = schema_file_path
                if not os.path.isabs(schema_full_file_path):
                    schema_full_file_path = str(Path(settings.BASE_DIR) / schema_full_file_path)

                with open(schema_full_file_path, "r") as f:
                    attached_files[schema_file_simple_path] = f.read()
                    logger.info("Mounted schema %s to VFS", schema_file_simple_path)
            except Exception as exp:
                raise Exception("run_deep_agent_on_session | error loading schema file: %s", exp)


def run_deep_agent_on_session(
        prompt_key: str,
        rerun_prompt_key: Optional[str],
        recovery_prompt_key: Optional[str],
        session: Session,
        output_file_name: str,
        additional_attached_files: Optional[Dict[str, str]] = None,
        run_main_step_text: str = "Running Agent ...",
        update_task_callback=None,
        force_run: bool = False,
        force_clean_run: bool = False,
        additional_parameters: Optional[Dict[str, Any]] = None,
        auto_update_session: bool = True,
        extra_tools: Optional[List] = None,
        quality_check_prompt_key=None,
        schema_file_path: Optional[str] = None,
        custom_validator: Optional[CustomLLMResponseValidator] = None
):
    if session.is_agent_finished and not force_run:
        return {"result": session.agent_result}, None
    try:
        # Always prefix output file name with '/' to ensure it's treated as an absolute path in the agent's VFS, even if the caller passed a relative name.
        if not output_file_name.startswith("/"):
            output_file_name = "/" + output_file_name

        file_content = get_files_summary(session)

        kb_manifest = generate_kb_manifest(session)
        # noinspection PyTypeChecker
        attached_files = prepare_attached_files_for_session(session)

        if additional_attached_files:
            attached_files = {**additional_attached_files, **attached_files}

        # Detect refinement mode: prev_agent_result is set when the user clicked "Refine".
        # In refinement mode, pre-seed the agent VFS with the previous result and all
        # existing output files so the agent can read and revise them.
        is_refinement = bool(session.prev_agent_result) and not force_clean_run
        if is_refinement:
            refinement_files: dict[str, str] = {}
            # noinspection PyUnresolvedReferences
            for f in session.attached_files.filter(
                    file_type=AttachedFile.FILE_TYPE_OUTPUT
            ):
                file_text_content = get_text_content(f)
                if file_text_content:
                    refinement_files[f.name if f.name.startswith("/") else "/" + f.name] = file_text_content

            # Always expose the previous report as 'output_file_name' (may already be there as hidden file)
            refinement_files[output_file_name] = session.prev_agent_result
            # Refinement output files take precedence over KB/input files
            attached_files = {**attached_files, **refinement_files}

        logger.info(
            "run_deep_agent_on_session | session=%s | seeding %d files",
            session.id,
            len(attached_files),
        )

        agent_tools, err = prepare_tools_for_session(session, update_task_callback)
        if err:
            return None, err

        agent_callbacks = None
        if update_task_callback:
            update_task_callback(progress_step=run_main_step_text)
            agent_callbacks = make_agent_callbacks(update_task_callback)

        agent_result, output_files = async_to_sync(run_deep_agent)(
            prompt_key=rerun_prompt_key if rerun_prompt_key and is_refinement else prompt_key,
            file_content=file_content,
            tools=agent_tools + (extra_tools or []),
            kb_manifest=kb_manifest,
            attached_files=attached_files,
            report_filename=output_file_name,
            recovery_prompt_key=recovery_prompt_key,
            agent_callbacks=agent_callbacks,
            additional_parameters=additional_parameters,
            quality_check_prompt_key=quality_check_prompt_key,
            schema_file_path=schema_file_path,
            custom_validator=custom_validator
        )

        if auto_update_session:
            session.agent_result = agent_result
            session.is_agent_finished = True
            session.save(
                update_fields=[
                    "agent_result",
                    "is_agent_finished",
                ]
            )

            output_files = {
                name: data for name, data in output_files.items()
                if additional_attached_files is None or data != additional_attached_files.get(name)
            }
            save_generation_output_files(session, output_files)

        return {"result": agent_result, "output_files": output_files}, None
    except PermissionError as error:
        logger.exception(
            "run_deep_agent_on_session | session=%s | permission error: %s",
            session.id,
            error,
        )
        return (
            None,
            "A permission error occurred while running deep agent. Please try again.",
        )
    except Exception as error:
        logger.exception(
            "run_deep_agent_on_session | session=%s | error: %s", session.id, error
        )
        return None, f"An error occurred while running deep agent: {repr(error)}"


def make_agent_callbacks(update_task_callback=None):
    """Build an AgentCallbacks instance wired to update_task_callback.

    When ``update_task_callback`` is supplied, all three callbacks are
    connected so that real-time agent progress (step tasks, thinking text,
    last tool call) is propagated to the SSE stream via the task row.
    When ``update_task_callback`` is ``None`` a no-op callbacks object is
    returned so callers that do not need streaming do not have to check for
    ``None``.
    """
    if update_task_callback is None:
        return None

    def on_thinking_update(thought: str):
        update_task_callback(thinking=thought)

    def on_tool_call(info: Dict[str, str]):
        tool = info.get("tool", "")

        # Ignore the todos update tool
        if 'todos' not in tool.lower():
            update_task_callback(last_tool_call=info)

    def on_todos_update(todos):
        if update_task_callback:
            update_task_callback(step_tasks=todos, thinking="", last_tool_call={})

    return AgentCallbacks(
        on_todos_update=on_todos_update,
        on_thinking_update=on_thinking_update,
        on_tool_call=on_tool_call,
    )


async def run_deep_agent(
        prompt_key: str,
        file_content: str,
        kb_manifest: str,
        report_filename: str,
        recovery_prompt_key: Optional[str] = None,
        attached_files: Optional[Dict[str, str]] = None,
        tools: Optional[List] = None,
        agent_callbacks=None,
        additional_parameters: Optional[Dict[str, Any]] = None,
        quality_check_prompt_key: Optional[str] = None,
        schema_file_path: Optional[str] = None,
        custom_validator: Optional[CustomLLMResponseValidator] = None
) -> tuple:
    """Use an agent with web or KB tools to execute a deep agent task and return results.

    Returns a ``(report_text, output_files_dict)`` tuple.

    ``report_text`` is the primary report (the content of ``report_filename``
    if written by the agent, otherwise the LLM response text).

    ``output_files_dict`` maps filenames (str) to content (str) for every file
    the agent wrote during this run, including ``report_filename``.

    ``tools`` must be an explicit non-empty list of LangChain tools to give the agent.
    Callers are responsible for constructing the correct tool set (web tools, KB RAG
    tools, or a combination) before invoking this function.
    """
    logger.info(
        "run_deep_agent | tools=%s | prompt_key=%s | attached_files=%s | report=%s",
        [t.name for t in tools] if tools else [],
        prompt_key,
        list(attached_files.keys()) if attached_files else [],
        report_filename,
    )
    try:
        output_files: dict = {}
        mount_schema_to_attached_files(schema_file_path, attached_files)
        attached_files = _process_input_files(attached_files)
        vfs_manifest = (
            "\n".join(f"- {p}" for p in attached_files.keys()) if attached_files else ""
        )
        template_params = {
            "current_date": str(datetime.datetime.now()),
            "file_content": file_content,
            "kb_manifest": f"KNOWLEDGE BASE INDEX:\n{kb_manifest}\n"
            if kb_manifest
            else "",
            "vfs_files": f"ATTACHED FILES:\n{vfs_manifest}\n"
            if vfs_manifest
            else "",
        }
        if additional_parameters:
            template_params.update(additional_parameters)
        result = await invoke_llm(
            prompt_key,
            system_message_template_name=f"{prompt_key}.system_message",
            user_message_template_name=f"{prompt_key}.prompt_template",
            template_params=template_params,
            is_agent=True,
            is_deep_agent=True,
            tools=tools,
            parse_json=False,
            agent_callbacks=agent_callbacks,
            output_files=output_files,
            attached_files=attached_files,
            fail_on_empty_response=not recovery_prompt_key
        )

        report_file_data: Optional[str] = output_files.get(report_filename)

        quality_pass, _, quality_fail_reason = await _perform_quality_check(
            result,
            quality_check_prompt_key,
            response_filename=report_filename,
            tools=tools,
            attached_files=attached_files,
            output_files=output_files,
            agent_callbacks=agent_callbacks,
            schema_file_path=schema_file_path,
            custom_validator=custom_validator
        )

        if quality_pass:
            assert report_file_data is not None
            logger.info(
                "run_deep_agent | using %s (%d chars)",
                report_filename,
                len(report_file_data),
            )
            return report_file_data, _process_output_files(output_files)
        else:
            if recovery_prompt_key:
                logger.warning(
                    f"run_deep_agent | quality check failed — "
                    f"Agent Quality check failed - {quality_fail_reason}. Retrying with attempt 1. "
                    f"LLM result={result}"
                )
                return await recover_deep_agent_run(
                    recovery_prompt_key=recovery_prompt_key,
                    report_filename=report_filename,
                    recovery_reason=quality_fail_reason,
                    tools=tools,
                    attached_files=attached_files,
                    recovered_files=output_files,
                    agent_callbacks=agent_callbacks,
                    quality_check_prompt_key=quality_check_prompt_key,
                    schema_file_path=schema_file_path,
                    custom_validator=custom_validator
                )
            else:
                logger.warning(
                    f"run_deep_agent | quality check failed — "
                    f"Agent Quality check failed - {quality_fail_reason}. No recovery prompt provided."
                    f"LLM result={result}"
                )
                raise Exception(
                    "run_deep_agent | agent did not pass the quality check and no recovery prompt provided"
                )
    except Exception as error:
        logger.exception("run_deep_agent | error: %s", error)
        raise


async def _perform_quality_check(
        llm_result,
        quality_check_prompt_key,
        response_filename,
        tools: Optional[List] = None,
        attached_files: Optional[Dict[str, str]] = None,
        output_files: Optional[Dict[str, str]] = None,
        agent_callbacks=None,
        schema_file_path: Optional[str] = None,
        custom_validator: Optional[CustomLLMResponseValidator] = None
):
    # Detect LLM Early-stop scenarios
    if llm_result is None or str(llm_result).strip() == "":
        return False, True, "Detected LLM Early-Stop or crash. LLM did not send a final response"

    if attached_files is None:
        attached_files = {}

    if output_files is None:
        output_files = {}

    merged_files = {**attached_files, **output_files}

    agent_result: Optional[str] = merged_files.get(response_filename)
    # If no result reject immediately

    recovery_report_found = agent_result and agent_result.strip()
    if not recovery_report_found:
        return False, True, f"Agent did not generate the necessary {response_filename}"

    # Perform schema validation first, if that fails reject
    if schema_file_path:
        assert agent_result is not None
        valid, err = validate_json_with_schema_file(agent_result, schema_file_path)
        if not valid:
            return False, True, f"Schema validation failed: {err}"

    if custom_validator:
        try:
            custom_validator.validate_response(
                agent_result,
                tools=tools,
                attached_files=attached_files,
                output_files=output_files,
                agent_callbacks=agent_callbacks,
            )
        except Exception as e:
            return False, True, f"LLM response validation failed: {e}"

    if quality_check_prompt_key:
        try:
            quality_check_result = await invoke_llm(
                quality_check_prompt_key,
                system_message_template_name=f"{quality_check_prompt_key}.system_message",
                user_message_template_name=f"{quality_check_prompt_key}.prompt_template",
                template_params={
                    "current_date": str(datetime.datetime.now()),
                    "agent_result": agent_result
                },
                is_agent=True,
                is_deep_agent=True,
                tools=tools,
                parse_json=True,
                agent_callbacks=agent_callbacks,
                attached_files=merged_files,
                fail_on_empty_response=True,
                 schema_file_path=str(Path(__file__).resolve().parent.parent / "resources" / "schema" / "quality_check_schema.json"),
            )

            quality_gate_passed = quality_check_result.get("is_approved", False)
            quality_gate_feedback = quality_check_result.get("feedback", "No feedback provided")
        except Exception as exp:
            logger.warning(f"Quality check prompt failed with error: {exp}")
            quality_gate_passed = False
            quality_gate_feedback = f"Quality check prompt failed with error: {exp}"

        if not quality_gate_passed:
            logger.warning(
                "_perform_quality_check | quality check failed: %s",
                quality_gate_feedback,
            )
    else:
        quality_gate_passed = True
        quality_gate_feedback = ""

    return quality_gate_passed, False, quality_gate_feedback


def _process_input_files(attached_files: dict[str, str] | None) -> dict[str, str] | None:
    return {
        path if is_text_file(path) else f'{path}.txt' if can_extract_text(path) else f'{path}.b64': content for
        path, content in attached_files.items()
    } if attached_files else attached_files


def _process_output_files(output_files: dict[str, str]) -> dict[str, str]:
    # If the output files has .EXT.txt, strip the .txt from it
    output_files_processed = {
        (filename[:-4] if (filename.endswith(".txt") or filename.endswith(".b64")) and len(
            filename) > 4 and '.' in filename[
                              :-4] else filename): content
        for filename, content in output_files.items()
    }
    return output_files_processed


async def recover_deep_agent_run(
        recovery_prompt_key: str,
        report_filename: str,
        recovery_reason: str,
        tools: Optional[List] = None,
        attached_files: Optional[Dict[str, str]] = None,
        recovered_files: Optional[Dict[str, str]] = None,
        agent_callbacks=None,
        recovery_attempt_num=1,
        quality_check_prompt_key=None,
        schema_file_path: Optional[str] = None,
        custom_validator: Optional[CustomLLMResponseValidator] = None
) -> tuple[str, Dict[str, str]]:
    """Attempt to recover a failed deep agent run by asking the agent to write the report again."""
    output_files: dict = {}

    if attached_files is None:
        attached_files = {}

    if recovered_files is None:
        recovered_files = {}

    mounted_files = {**attached_files, **recovered_files}

    result = await invoke_llm(
        recovery_prompt_key,
        system_message_template_name=f"{recovery_prompt_key}.system_message",
        user_message_template_name=f"{recovery_prompt_key}.prompt_template",
        template_params={
            "current_date": str(datetime.datetime.now()),
            "recovery_reason": recovery_reason
        },
        is_agent=True,
        is_deep_agent=True,
        tools=tools,
        parse_json=False,
        agent_callbacks=agent_callbacks,
        output_files=output_files,
        attached_files=mounted_files,
        fail_on_empty_response=False
    )
    merged_files = {**recovered_files, **output_files}
    if "/memory.md" in recovered_files and "/memory.md" in output_files:
        merged_files["/memory.md"] = (
                recovered_files["/memory.md"] + "\n" + output_files["/memory.md"]
        )

    recovery_report: Optional[str] = merged_files.get(report_filename)
    quality_pass, is_fatal, quality_fail_reason = await _perform_quality_check(
        result,
        quality_check_prompt_key,
        response_filename=report_filename,
        tools=tools,
        attached_files=attached_files,
        output_files=merged_files,
        agent_callbacks=agent_callbacks,
        schema_file_path=schema_file_path,
        custom_validator=custom_validator
    )

    if quality_pass:
        assert recovery_report is not None # Silence code check
        logger.info(
            "recover_deep_agent_run | auto-recovery succeeded (%d chars)",
            len(recovery_report),
        )
        return recovery_report, _process_output_files(merged_files)
    elif recovery_attempt_num < _RECOVERY_MAX_ATTEMPTS:
        logger.warning(
            f"recover_deep_agent_run | auto-recovery attempt {recovery_attempt_num} failed — "
            f"Agent Quality check failed - {quality_fail_reason}. Retrying with attempt {recovery_attempt_num + 1}. "
            f"LLM result={result}"
        )
        return await recover_deep_agent_run(
            recovery_prompt_key=recovery_prompt_key,
            report_filename=report_filename,
            recovery_reason=quality_fail_reason,
            tools=tools,
            attached_files=attached_files,
            recovered_files=merged_files,
            agent_callbacks=agent_callbacks,
            recovery_attempt_num=recovery_attempt_num + 1,
            quality_check_prompt_key=quality_check_prompt_key,
            schema_file_path=schema_file_path,
            custom_validator=custom_validator
        )
    elif not is_fatal:
        assert recovery_report is not None # Silence code check
        # Quality did not pass but we got a report back and retry count exceeded max
        # We will give up and return the result anyway at this point
        logger.warning(
            f"recover_deep_agent_run | auto-recovery attempt {recovery_attempt_num} failed — "
            f"Agent Quality check failed - {quality_fail_reason}. Will return the result anyway since max attempts exceeded. "
            f"LLM result={result}"
        )
        return recovery_report, _process_output_files(merged_files)
    else:
        logger.warning(
            f"recover_deep_agent_run | auto-recovery failed — agent did not write {report_filename}. LLM result={result}"
        )
        raise RuntimeError("agent did not pass quality checks after recovery attempt")


def save_generation_output_files(
        session,
        output_files: dict[str, Union[str, bytes]],
        always_hidden_filenames: set[str] | None = None,
        append_to_existing: set[str] | None = None,
        ignore_existing: bool = False,
        offload_immediately: bool = False,
):
    """Persist agent-generated files as output AttachedFile records.

    Args:
        session: The Session owning these output files.
        output_files: Dict of {filename: content} for files written by the agent.
        always_hidden_filenames: Filenames that are always saved as hidden (e.g., {"report.md", "memory.md"}).
        append_to_existing: Filenames where new content is appended to existing record instead of replacing.
        ignore_existing: If True, skip saving any file that already exists on the session instead of replacing or appending.
        offload_immediately: Offload any data to cold storage immediately
    """
    if always_hidden_filenames is None:
        always_hidden_filenames = {"report.md", "memory.md"}
    if append_to_existing is None:
        append_to_existing = {"memory.md"}

    # Filter the files in case they start with 'kb' or 'docs' or '/kb' or '/docs',
    # which are reserved for input files and should not be overwritten by agent output.
    output_files = {
        filename: content
        for filename, content in output_files.items()
        if not (filename.startswith("kb"))
           and not filename.startswith("docs")
           and not filename.startswith("/kb")
           and not filename.startswith("/docs")
    }

    new_files: dict[str, str | bytes] = {}
    for filename, content in output_files.items():
        if filename and filename.strip() and content and str(content).strip():
            new_files[filename.lstrip("/")] = content

    # If 'ignore_existing' is True, we filter out any existing output file on the session
    if ignore_existing:
        existing_output_files = session.attached_files.filter(
            file_type=AttachedFile.FILE_TYPE_OUTPUT
        ).values_list("name", flat=True)
        new_files = {
            name: content
            for name, content in new_files.items()
            if name not in existing_output_files
        }

    files_to_create = []
    for clean_name, content in new_files.items():
        is_hidden = clean_name.lower() in always_hidden_filenames

        if clean_name in append_to_existing:
            existing = AttachedFile.objects.filter(
                session=session,
                file_type=AttachedFile.FILE_TYPE_OUTPUT,
                name=clean_name,
            ).first()
            if existing:
                if isinstance(content, bytes):
                    raise Exception(
                        f"Appending to existing files is only supported for text content, but got bytes for file {clean_name}")

                existing_text = get_text_content(existing)
                content = f"{existing_text}\n\n---\n\n{content}"

        af = AttachedFile(
            session=session,
            file_type=AttachedFile.FILE_TYPE_OUTPUT,
            name=clean_name,
            is_hidden=is_hidden,
            binary_content=_parse_content_to_raw_data(content),
        )
        files_to_create.append(af)

    if files_to_create:
        # Delete all existing files
        AttachedFile.objects.filter(
            session=session,
            file_type=AttachedFile.FILE_TYPE_OUTPUT,
            name__in=set([af.name for af in files_to_create]),
        ).delete()

        # Recreate the files with the new content
        for file_obj in files_to_create:
            binary_content = file_obj.binary_content
            file_obj.binary_content = None
            file_obj.save()
            set_binary_content(file_obj, binary_content, offload_immediately=offload_immediately)

        logger.info(
            "save_generation_output_files | session=%s | saved %d output files",
            session.id,
            len(files_to_create),
        )


def _parse_content_to_raw_data(content) -> bytes:
    if isinstance(content, bytes):
        raw_data = content
    elif isinstance(content, str):
        if is_base64(content):
            raw_data = base64.decodebytes(content.encode("utf-8"))
        else:
            raw_data = content.encode("utf-8")
    else:
        raw_data = str(content).encode("utf-8")

    return raw_data


def generate_kb_manifest(session) -> str:
    """Build KB manifest from attached KnowledgeBases."""

    kb_manifest_lines = []
    unauthorized_kbs = session.knowledge_bases.exclude(user=session.user)
    if unauthorized_kbs.exists():
        raise PermissionError(
            f"Session {session.id} has knowledge bases belonging to another user: "
            f"{list(unauthorized_kbs.values_list('id', flat=True))}"
        )
    for kb in session.knowledge_bases.prefetch_related("kb_files").all():
        for f in kb.kb_files.all():
            f_sum = get_file_summary(f)
            if f_sum:
                kb_manifest_lines.append(
                    f"- /kb/{kb.name}/{f.name.lstrip('/')}: {f_sum}"
                )
            else:
                kb_manifest_lines.append(f"- /kb/{kb.name}/{f.name.lstrip('/')}")

    kb_manifest = "\n".join(kb_manifest_lines)
    return kb_manifest


def prepare_attached_files_for_session(session) -> dict[str, str]:
    """Build VFS files from KB files, session input attached files, and attached sessions."""
    kb_vfs_files: Dict[str, str] = {}
    for kb in session.knowledge_bases.prefetch_related("kb_files").all():
        for f in kb.kb_files.all():
            text = get_text_content(f)
            if text:
                path = f"/kb/{kb.name}/{f.name.lstrip('/')}"
                kb_vfs_files[path] = text
    input_vfs_files: Dict[str, str] = {}
    for f in session.attached_files.filter(file_type=AttachedFile.FILE_TYPE_INPUT):
        text = get_text_content(f)
        if text:
            path = f"/docs/{f.name.lstrip('/')}"
            input_vfs_files[path] = text
    for attached_session in session.attached_sessions.all():
        if attached_session.user != session.user and not attached_session.is_public:
            continue
        session_file, output_files = parse_session_to_file(attached_session)
        session_text = get_text_content(session_file)
        if session_text:
            path = f"/docs/{session_file.name}"
            input_vfs_files[path] = session_text
        for output_file in output_files:
            output_text = get_text_content(output_file)
            if output_text:
                prefixed_name = f"/session_{attached_session.id}/{output_file.name.lstrip('/')}"
                input_vfs_files[prefixed_name] = output_text
    return {**kb_vfs_files, **input_vfs_files}


def prepare_tools_for_session(session: Session, update_task_callback=None):
    """Prepare tools for a session: KB RAG tools, MCP servers, web search.

    Returns (tools, error) tuple. Tools is a list of LangChain tools, error is None on success.
    """
    kb_files, err = preprocess_kb_files(session, update_task_callback)
    if err:
        return None, err

    kb_rag_tools = build_kb_rag_tools(kb_files) if kb_files else []

    cfg = LLMConfiguration.get_solo()

    mcp_tools: list = []
    if session.mcp_servers.exists():
        unauthorized = session.mcp_servers.exclude(user=session.user)
        if unauthorized.exists():
            raise PermissionError(
                f"Session {session.id} has MCP servers belonging to another user: "
                f"{list(unauthorized.values_list('id', flat=True))}"
            )
        if update_task_callback:
            update_task_callback(progress_step="Connecting MCP Servers ...")

        try:
            mcp_tools = build_mcp_tools_from_servers(list(session.mcp_servers.all()))
        except Exception as exp:
            logger.exception("prepare_tools_for_session | error loading MCP tools: %s", exp)
            mcp_tools = []

    session_tools = []
    if session.is_research_online:
        session_tools.extend(get_web_tools())

        if session.user.is_pro and cfg.enable_playwright:
            session_tools.extend(get_playwright_tools())

    tunnel_tools = []
    if session.desktop_tunnel_servers:
        tunnel_ids = list(session.desktop_tunnel_servers.keys())
        unauthorized = Tunnel.objects.filter(tunnel_id__in=tunnel_ids).exclude(user=session.user)
        if unauthorized.exists():
            raise PermissionError(
                f"Session {session.id} has Tunnels belonging to another user: "
                f"{list(unauthorized.values_list('tunnel_id', flat=True))}"
            )

        tunnel_tools = build_tunnel_tools(
            tunnel_ids,
            session.desktop_tunnel_servers
        )

    subagent_tools = [spawn_subagent] if cfg.enable_subagents else []
    all_tools = subagent_tools + get_deepagent_validation_tools() + session_tools + kb_rag_tools + mcp_tools + tunnel_tools

    return all_tools, None
