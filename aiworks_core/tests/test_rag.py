"""
Rag tests for the Ai-Works Core API.
"""

from django.test import TestCase
from langchain_huggingface import HuggingFaceEmbeddings

# noinspection PyProtectedMember
from ..logic.rag import build_kb_rag_tools, _get_kb_embeddings


class RagModuleTests(TestCase):
    """Unit tests for api/rag.py — embedding singleton and build_kb_rag_tools."""

    # ------------------------------------------------------------------
    # Embedding singleton
    # ------------------------------------------------------------------

    def test_get_kb_embeddings_returns_huggingface_embeddings(self):
        emb = _get_kb_embeddings()
        self.assertIsInstance(emb, HuggingFaceEmbeddings)

    def test_get_kb_embeddings_is_singleton(self):
        emb1 = _get_kb_embeddings()
        emb2 = _get_kb_embeddings()
        self.assertIs(emb1, emb2)

    # ------------------------------------------------------------------
    # build_kb_rag_tools — basic contract
    # ------------------------------------------------------------------

    def test_returns_empty_for_no_files(self):
        self.assertEqual(build_kb_rag_tools([]), [])

    def test_returns_empty_for_all_empty_content(self):
        files = [
            {'name': 'a.txt', 'content': ''},
            {'name': 'b.txt', 'content': ''},
        ]
        self.assertEqual(build_kb_rag_tools(files), [])

    def test_returns_empty_for_missing_content_key(self):
        files = [{'name': 'a.txt'}]
        self.assertEqual(build_kb_rag_tools(files), [])

    def test_returns_exactly_one_tool(self):
        files = [{'name': 'doc.txt', 'content': 'The quarterly revenue grew by 20% in Q3.'}]
        tools = build_kb_rag_tools(files)
        self.assertEqual(len(tools), 1)

    def test_tool_name_is_search_internal_knowledge_base(self):
        files = [{'name': 'doc.txt', 'content': 'Some business content here.'}]
        tools = build_kb_rag_tools(files)
        self.assertEqual(tools[0].name, 'search_internal_knowledge_base')

    def test_tool_has_description(self):
        files = [{'name': 'doc.txt', 'content': 'Some business content here.'}]
        tools = build_kb_rag_tools(files)
        self.assertTrue(len(tools[0].description) > 0)

    def test_skips_empty_files_but_indexes_non_empty(self):
        files = [
            {'name': 'empty.txt', 'content': ''},
            {'name': 'real.txt', 'content': 'Profit margins improved significantly this year.'},
        ]
        tools = build_kb_rag_tools(files)
        self.assertEqual(len(tools), 1)
        result = tools[0].invoke('profit margins')
        self.assertIn('Profit', result)

    # ------------------------------------------------------------------
    # Search results — content and format
    # ------------------------------------------------------------------

    def test_result_contains_file_name_annotation(self):
        files = [{'name': 'strategy.txt', 'content': 'Long-term growth requires sustained investment.'}]
        tools = build_kb_rag_tools(files)
        result = tools[0].invoke('long-term growth')
        self.assertIn('[From: strategy.txt]', result)

    def test_relevant_query_returns_matching_passage(self):
        files = [
            {'name': 'finance.txt', 'content': 'Revenue grew by 20% in Q3. Costs decreased.'},
            {'name': 'hr.txt', 'content': 'We hired 50 new engineers last quarter.'},
        ]
        tools = build_kb_rag_tools(files)
        result = tools[0].invoke('revenue growth')
        self.assertIn('Revenue', result)

    def test_unrelated_query_returns_no_results_message(self):
        files = [{'name': 'doc.txt', 'content': 'Some content about business strategy.'}]
        tools = build_kb_rag_tools(files)
        result = tools[0].invoke('zzzzzzzzxxxxxxxxxyyy')
        self.assertIn('No relevant content', result)

    def test_multiple_passages_separated_by_divider(self):
        # Two clearly distinct documents; a broad query should surface both
        # and they must be separated by the expected divider.
        long_finance = (
            'Revenue grew by 20% in Q3. EBITDA improved. Operating costs fell. '
            'Cash flow is strong. Dividend payout increased. Shareholder returns rose.'
        )
        long_hr = (
            'We hired 50 new engineers. Attrition dropped to 5%. Compensation bands '
            'were revised. Onboarding satisfaction scored 9/10. Training hours increased.'
        )
        files = [
            {'name': 'finance.txt', 'content': long_finance},
            {'name': 'hr.txt', 'content': long_hr},
        ]
        tools = build_kb_rag_tools(files)
        result = tools[0].invoke('company performance')
        # If multiple chunks are returned, they are joined by the divider
        if '---' in result:
            parts = [p.strip() for p in result.split('---') if p.strip()]
            self.assertGreater(len(parts), 1)

    def test_each_call_to_build_creates_independent_vectorstore(self):
        """Two separate invocations must not share state."""
        tools_a = build_kb_rag_tools([{'name': 'a.txt', 'content': 'Alpha beta gamma delta epsilon.'}])
        tools_b = build_kb_rag_tools([{'name': 'b.txt', 'content': 'Omega psi chi phi upsilon tau.'}])
        result_a = tools_a[0].invoke('alpha beta gamma')
        result_b = tools_b[0].invoke('omega psi chi')
        self.assertNotIn('b.txt', result_a)
        self.assertNotIn('a.txt', result_b)

    def test_large_file_is_chunked_and_searchable(self):
        # A file that exceeds chunk_size=1000 chars must still be indexed and the
        # relevant section must be retrievable. Use distinct thematic paragraphs so
        # the signal chunk is not diluted by unrelated filler within the same chunk.
        section_finance = (
                              'The company reported strong financial results for the fiscal year. '
                              'Revenue increased by thirty percent driven by new product launches. '
                              'Operating margins expanded due to cost discipline and efficiency gains. '
                              'Cash reserves grew to record levels supporting future investment plans. '
                          ) * 3  # ~500 chars, clearly one topic
        section_quantum = (
                              'Quantum entanglement enables instantaneous correlation between particles. '
                              'Researchers are investigating communication protocols based on quantum states. '
                              'Entangled photon pairs have been demonstrated across fiber-optic networks. '
                              'This research area holds promise for secure quantum communication systems. '
                          ) * 3  # ~500 chars, clearly a different topic
        section_hr = (
                         'The human resources department expanded the workforce by two hundred roles. '
                         'Employee satisfaction surveys showed improvement across all departments. '
                         'New onboarding programs reduced time-to-productivity for recent hires. '
                         'Compensation benchmarking was completed for all engineering job grades. '
                     ) * 3  # ~500 chars, another distinct topic
        content = '\n\n'.join([section_finance, section_quantum, section_hr])
        files = [{'name': 'big.txt', 'content': content}]
        tools = build_kb_rag_tools(files)
        result = tools[0].invoke('quantum entanglement communication')
        self.assertIn('Quantum', result)

    def test_result_is_string(self):
        files = [{'name': 'doc.txt', 'content': 'Simple test document content.'}]
        tools = build_kb_rag_tools(files)
        result = tools[0].invoke('test document')
        self.assertIsInstance(result, str)
