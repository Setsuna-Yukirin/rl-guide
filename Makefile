# rl-guide Makefile

.PHONY: help test test-unit test-integration lint format clean install

# 默认目标
help:
	@echo "rl-guide 开发命令"
	@echo ""
	@echo "可用命令:"
	@echo "  make test          - 运行所有测试"
	@echo "  make test-unit     - 运行单元测试"
	@echo "  make test-int      - 运行集成测试"
	@echo "  make lint          - 代码检查"
	@echo "  make format        - 代码格式化"
	@echo "  make clean         - 清理临时文件"
	@echo "  make install       - 安装依赖"
	@echo ""

# 运行所有测试
test:
	@echo "运行所有测试..."
	pytest tests/ -v --tb=short

# 运行单元测试
test-unit:
	@echo "运行单元测试..."
	pytest tests/ -m "unit" -v --tb=short

# 运行集成测试
test-int:
	@echo "运行集成测试..."
	pytest tests/ -m "integration" -v --tb=short

# 代码检查
lint:
	@echo "运行代码检查..."
	black --check utils/ tests/
	pylint utils/ tests/ --rcfile=.pylintrc || true

# 代码格式化
format:
	@echo "格式化代码..."
	black utils/ tests/

# 清理临时文件
clean:
	@echo "清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -delete
	find . -type d -name "checkpoints" -delete
	@echo "清理完成"

# 安装依赖
install:
	@echo "安装依赖..."
	pip install -r requirements.txt
	pip install pytest pytest-cov black pylint
	@echo "安装完成"

# 快速测试（用于开发）
test-fast:
	pytest tests/ -m "unit" -x -q

# 生成覆盖率报告
coverage:
	pytest tests/ --cov=utils --cov-report=html
	@echo "覆盖率报告已生成：htmlcov/index.html"
