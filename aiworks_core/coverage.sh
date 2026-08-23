#!/bin/bash
#
# coverage.sh - Generate code coverage report for App backend tests
#
# This script runs the test suite with coverage analysis and generates a detailed report.
# It supports multiple output formats and provides summary statistics.
#
# Usage:
#   ./coverage.sh [options]
#
# Options:
#   --html    Generate HTML coverage report (saved to htmlcov/)
#   --xml     Generate XML coverage report (saved to coverage.xml)
#   --json    Generate JSON coverage report (saved to coverage.json)
#   --all     Generate all report formats
#   --help    Show this help message
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
GENERATE_HTML=false
GENERATE_XML=false
GENERATE_JSON=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --html)
            GENERATE_HTML=true
            shift
            ;;
        --xml)
            GENERATE_XML=true
            shift
            ;;
        --json)
            GENERATE_JSON=true
            shift
            ;;
        --all)
            GENERATE_HTML=true
            GENERATE_XML=true
            GENERATE_JSON=true
            shift
            ;;
        --help)
            head -n 18 "$0" | tail -n +3 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to the backend directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  App - Coverage${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check if coverage is installed
if ! python3 -c "import coverage" 2>/dev/null; then
    echo -e "${YELLOW}Coverage.py is not installed. Installing...${NC}"
    pip install coverage pytest-cov
    echo
fi

# Clean previous coverage data
echo -e "${YELLOW}Cleaning previous coverage data...${NC}"
rm -f .coverage coverage.xml coverage.json
rm -rf htmlcov/
echo

# Run tests with coverage
echo -e "${GREEN}Running tests with coverage analysis...${NC}"
echo
if python3 -m coverage run --source='api' manage.py test api; then
    echo
    echo -e "${GREEN}✓ Tests completed successfully${NC}"
else
    echo
    echo -e "${RED}✗ Tests failed${NC}"
    exit 1
fi

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Coverage Report${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Generate coverage report to console
python3 -m coverage report

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Missing Coverage Details${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Show lines that are not covered
python3 -m coverage report --show-missing

# Generate additional report formats if requested
if [ "$GENERATE_HTML" = true ]; then
    echo
    echo -e "${YELLOW}Generating HTML coverage report...${NC}"
    python3 -m coverage html
    echo -e "${GREEN}✓ HTML report generated in htmlcov/index.html${NC}"
fi

if [ "$GENERATE_XML" = true ]; then
    echo
    echo -e "${YELLOW}Generating XML coverage report...${NC}"
    python3 -m coverage xml
    echo -e "${GREEN}✓ XML report generated as coverage.xml${NC}"
fi

if [ "$GENERATE_JSON" = true ]; then
    echo
    echo -e "${YELLOW}Generating JSON coverage report...${NC}"
    python3 -m coverage json
    echo -e "${GREEN}✓ JSON report generated as coverage.json${NC}"
fi

echo
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Coverage Analysis Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo

# Show summary statistics
COVERAGE_PERCENT=$(python3 -m coverage report | grep TOTAL | awk '{print $4}')
echo -e "${BLUE}Overall Coverage: ${GREEN}${COVERAGE_PERCENT}${NC}"

if [ "$GENERATE_HTML" = true ]; then
    echo -e "${BLUE}View detailed report: ${YELLOW}htmlcov/index.html${NC}"
fi

echo
