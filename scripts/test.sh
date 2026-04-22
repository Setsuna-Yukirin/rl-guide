#!/bin/bash

# 测试脚本

set -e

echo "======================================"
echo "rl-guide 测试套件"
echo "======================================"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 统计
TESTS_PASSED=0
TESTS_FAILED=0

# 1. 代码格式化检查
echo -e "\n${YELLOW}[1/4] 代码格式化检查 (Black)...${NC}"
if black --check utils/ tests/ 2>/dev/null; then
    echo -e "${GREEN}✓ 代码格式化通过${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ 代码格式化失败，运行 'black utils/ tests/' 修复${NC}"
    ((TESTS_FAILED++))
fi

# 2. 运行单元测试
echo -e "\n${YELLOW}[2/4] 运行单元测试...${NC}"
if pytest tests/ -m "not slow" -v --tb=short; then
    echo -e "${GREEN}✓ 单元测试通过${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ 单元测试失败${NC}"
    ((TESTS_FAILED++))
fi

# 3. 运行集成测试
echo -e "\n${YELLOW}[3/4] 运行集成测试...${NC}"
if pytest tests/ -m "integration" -v --tb=short; then
    echo -e "${GREEN}✓ 集成测试通过${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ 集成测试失败${NC}"
    ((TESTS_FAILED++))
fi

# 4. 代码覆盖率
echo -e "\n${YELLOW}[4/4] 代码覆盖率检查...${NC}"
if pytest tests/ --cov=utils --cov-report=term-missing --cov-fail-under=70 2>/dev/null; then
    echo -e "${GREEN}✓ 代码覆盖率达标 (70%+)${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠ 代码覆盖率未达标或 coverage 未安装${NC}"
fi

# 总结
echo -e "\n======================================"
echo -e "测试总结:"
echo -e "  通过：${GREEN}${TESTS_PASSED}${NC}"
echo -e "  失败：${RED}${TESTS_FAILED}${NC}"
echo "======================================"

if [ $TESTS_FAILED -gt 0 ]; then
    exit 1
else
    echo -e "${GREEN}✅ 所有测试通过！${NC}"
    exit 0
fi
