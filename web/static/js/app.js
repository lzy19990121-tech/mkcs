/**
 * MKCS Web 前端交互优化
 * 提供：
 * - 表��排序
 * - 表格筛选
 * - 快捷操作
 * - 响应式布局
 */

// ==================== 表格排序功能 ====================

class SortableTable {
    constructor(tableElement) {
        this.table = tableElement;
        this.currentSortColumn = -1;
        this.currentSortDirection = 'asc';
        this.init();
    }

    init() {
        const headers = this.table.querySelectorAll('th[data-sortable]');
        headers.forEach((header, index) => {
            header.style.cursor = 'pointer';
            header.style.position = 'relative';
            header.addEventListener('click', () => this.sortByColumn(index));
        });
    }

    sortByColumn(columnIndex) {
        const tbody = this.table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));

        // 切换排序方向
        if (this.currentSortColumn === columnIndex) {
            this.currentSortDirection = this.currentSortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.currentSortColumn = columnIndex;
            this.currentSortDirection = 'asc';
        }

        // 更��排序指示器
        this.updateSortIndicators();

        // 排序
        rows.sort((a, b) => {
            const aValue = this.getCellValue(a, columnIndex);
            const bValue = this.getCellValue(b, columnIndex);
            return this.compareValues(aValue, bValue);
        });

        // 重新插入
        rows.forEach(row => tbody.appendChild(row));
    }

    getCellValue(row, columnIndex) {
        const cell = row.cells[columnIndex];
        const text = cell.textContent.trim();

        // 尝试解析数字
        if (text.startsWith('$') || text.startsWith('+') || text.startsWith('-')) {
            const num = parseFloat(text.replace(/[$,+%]/g, ''));
            if (!isNaN(num)) return num;
        }

        // 尝试解析百分比
        if (text.endsWith('%')) {
            const num = parseFloat(text.replace('%', ''));
            if (!isNaN(num)) return num;
        }

        return text;
    }

    compareValues(a, b) {
        let result = 0;

        if (typeof a === 'number' && typeof b === 'number') {
            result = a - b;
        } else {
            result = String(a).localeCompare(String(b));
        }

        return this.currentSortDirection === 'asc' ? result : -result;
    }

    updateSortIndicators() {
        const headers = this.table.querySelectorAll('th');
        headers.forEach((header, index) => {
            const indicator = header.querySelector('.sort-indicator');
            if (indicator) indicator.remove();

            if (index === this.currentSortColumn) {
                const arrow = this.currentSortDirection === 'asc' ? '↑' : '↓';
                header.innerHTML += ` <span class="sort-indicator">${arrow}</span>`;
            }
        });
    }
}

// ==================== 表格筛选功能 ====================

class FilterableTable {
    constructor(tableElement, filterInput) {
        this.table = tableElement;
        this.filterInput = filterInput;
        this.originalRows = Array.from(tableElement.querySelectorAll('tbody tr'));
        this.init();
    }

    init() {
        this.filterInput.addEventListener('input', (e) => this.filter(e.target.value));
        this.filterInput.addEventListener('keyup', (e) => {
            if (e.key === 'Escape') {
                this.filterInput.value = '';
                this.filter('');
            }
        });
    }

    filter(query) {
        const lowerQuery = query.toLowerCase();
        const tbody = this.table.querySelector('tbody');
        tbody.innerHTML = '';

        this.originalRows.forEach(row => {
            const rowText = row.textContent.toLowerCase();
            const matches = rowText.includes(lowerQuery);
            row.style.display = matches ? '' : 'none';
            tbody.appendChild(row);
        });

        // 更新计数
        this.updateCount();
    }

    updateCount() {
        const visibleRows = this.originalRows.filter(row => row.style.display !== 'none');
        const countElement = this.table.parentElement.querySelector('.filter-count');
        if (countElement) {
            countElement.textContent = `显示 ${visibleRows.length} / ${this.originalRows.length} 条`;
        }
    }
}

// ==================== 快捷操作功能 ====================

class QuickActions {
    constructor() {
        this.actions = new Map();
        this.init();
    }

    init() {
        // 键盘快捷键
        document.addEventListener('keydown', (e) => this.handleKeyPress(e));
    }

    register(key, callback, description) {
        this.actions.set(key, { callback, description });
    }

    handleKeyPress(e) {
        // Ctrl/Cmd + Key
        if (e.ctrlKey || e.metaKey) {
            const action = this.actions.get(e.key.toLowerCase());
            if (action) {
                e.preventDefault();
                action.callback();
            }
        }
    }

    showHelp() {
        const help = Array.from(this.actions.entries())
            .map(([key, { description }]) => `Ctrl+${key.toUpperCase()}: ${description}`)
            .join('\n');
        alert('快捷键:\n' + help);
    }
}

// ==================== 数据导出功能 ====================

class TableExporter {
    constructor(tableElement) {
        this.table = tableElement;
    }

    exportCSV(filename) {
        const rows = Array.from(this.table.querySelectorAll('tr'));
        const csv = rows.map(row => {
            const cells = Array.from(row.querySelectorAll('th, td'));
            return cells.map(cell => {
                const text = cell.textContent.trim().replace(/"/g, '""');
                return `"${text}"`;
            }).join(',');
        }).join('\n');

        this.downloadFile(csv, filename, 'text/csv');
    }

    exportJSON(filename) {
        const headers = Array.from(this.table.querySelectorAll('th')).map(th => th.textContent.trim());
        const rows = Array.from(this.table.querySelectorAll('tbody tr'));

        const data = rows.map(row => {
            const cells = Array.from(row.querySelectorAll('td'));
            const obj = {};
            headers.forEach((header, i) => {
                obj[header] = cells[i]?.textContent.trim() || '';
            });
            return obj;
        });

        this.downloadFile(JSON.stringify(data, null, 2), filename, 'application/json');
    }

    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }
}

// ==================== 响应式表格功能 ====================

class ResponsiveTable {
    constructor(tableElement) {
        this.table = tableElement;
        this.init();
    }

    init() {
        // 添加响应式类
        this.table.classList.add('responsive-table');

        // 监听窗口大小变化
        window.addEventListener('resize', () => this.adjustColumns());
        this.adjustColumns();

        // 添加列显示/隐藏控制
        this.createColumnToggle();
    }

    adjustColumns() {
        const width = window.innerWidth;
        const thead = this.table.querySelector('thead');
        if (!thead) return;

        const headers = Array.from(thead.querySelectorAll('th'));

        // 小屏幕隐藏次要列
        if (width < 768) {
            headers.forEach((th, i) => {
                const isImportant = th.dataset.important === 'true';
                this.toggleColumn(i, isImportant);
            });
        } else {
            headers.forEach((_, i) => this.toggleColumn(i, true));
        }
    }

    toggleColumn(index, show) {
        const rows = this.table.querySelectorAll('tr');
        rows.forEach(row => {
            if (row.cells[index]) {
                row.cells[index].style.display = show ? '' : 'none';
            }
        });
    }

    createColumnToggle() {
        const container = this.table.parentElement;
        const toolbar = document.createElement('div');
        toolbar.className = 'table-toolbar';

        const headers = this.table.querySelectorAll('th');
        const toggles = Array.from(headers).map((th, i) => {
            const label = document.createElement('label');
            label.innerHTML = `
                <input type="checkbox" checked data-column="${i}">
                ${th.textContent.trim()}
            `;
            label.querySelector('input').addEventListener('change', (e) => {
                this.toggleColumn(i, e.target.checked);
            });
            return label;
        });

        toolbar.appendChild(document.createElement('div')).className = 'column-toggles';
        toolbar.querySelector('.column-toggles').append(...toggles);

        container.insertBefore(toolbar, this.table);
    }
}

// ==================== 交易详情模态框 ====================

class TradeDetailModal {
    constructor() {
        this.modal = null;
        this.init();
    }

    init() {
        // 创建模态框
        this.modal = document.createElement('div');
        this.modal.className = 'trade-modal';
        this.modal.innerHTML = `
            <div class="modal-overlay">
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>交易详情</h3>
                        <button class="modal-close">&times;</button>
                    </div>
                    <div class="modal-body"></div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary modal-close-btn">关闭</button>
                        <button class="btn btn-primary view-replay-btn">查看复盘</button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(this.modal);

        // 绑定关闭事件
        this.modal.querySelectorAll('.modal-close, .modal-close-btn').forEach(btn => {
            btn.addEventListener('click', () => this.close());
        });

        this.modal.querySelector('.modal-overlay').addEventListener('click', (e) => {
            if (e.target === e.currentTarget) this.close();
        });
    }

    show(tradeData) {
        const body = this.modal.querySelector('.modal-body');
        body.innerHTML = this.renderTradeDetail(tradeData);
        this.modal.style.display = 'flex';

        // 绑定复盘按钮
        const replayBtn = this.modal.querySelector('.view-replay-btn');
        replayBtn.onclick = () => {
            this.close();
            this.viewReplay(tradeData);
        };
    }

    close() {
        this.modal.style.display = 'none';
    }

    renderTradeDetail(trade) {
        return `
            <div class="trade-detail-grid">
                <div class="detail-row">
                    <span class="label">交易 ID:</span>
                    <span class="value">${trade.trade_id || '-'}</span>
                </div>
                <div class="detail-row">
                    <span class="label">标的:</span>
                    <span class="value">${trade.symbol}</span>
                </div>
                <div class="detail-row">
                    <span class="label">方向:</span>
                    <span class="value badge ${trade.side.toLowerCase()}">${trade.side}</span>
                </div>
                <div class="detail-row">
                    <span class="label">价格:</span>
                    <span class="value">$${trade.price?.toFixed(2) || '-'}</span>
                </div>
                <div class="detail-row">
                    <span class="label">数量:</span>
                    <span class="value">${trade.quantity || '-'}</span>
                </div>
                <div class="detail-row">
                    <span class="label">盈亏:</span>
                    <span class="value ${trade.pnl > 0 ? 'positive' : trade.pnl < 0 ? 'negative' : ''}">
                        ${trade.pnl ? (trade.pnl > 0 ? '+' : '') + '$' + trade.pnl.toFixed(2) : '-'}
                    </span>
                </div>
                <div class="detail-row">
                    <span class="label">时间:</span>
                    <span class="value">${new Date(trade.timestamp).toLocaleString('zh-CN')}</span>
                </div>
                <div class="detail-row">
                    <span class="label">策略:</span>
                    <span class="value">${trade.strategy || '-'}</span>
                </div>
                <div class="detail-row full-width">
                    <span class="label">信号上下文:</span>
                    <pre class="value">${JSON.stringify(trade.signal_context || {}, null, 2)}</pre>
                </div>
            </div>
        `;
    }

    viewReplay(trade) {
        // 跳转到复盘页面
        const params = new URLSearchParams({
            trade_id: trade.trade_id,
            symbol: trade.symbol,
            timestamp: trade.timestamp
        });
        window.location.href = `/replay?${params.toString()}`;
    }
}

// ==================== 初始化所有功能 ====================

function initTableEnhancements() {
    // 初始化所有可排序表格
    document.querySelectorAll('table[data-sortable="true"]').forEach(table => {
        new SortableTable(table);
    });

    // 初始化所有可筛选表格
    document.querySelectorAll('[data-filter-target]').forEach(input => {
        const targetId = input.dataset.filterTarget;
        const table = document.getElementById(targetId);
        if (table) {
            new FilterableTable(table, input);
        }
    });

    // 初始化快捷操作
    const quickActions = new QuickActions();
    quickActions.register('e', () => exportCurrentTable(), '导出表格');
    quickActions.register('f', () => focusFilterInput(), '聚焦筛选框');
    quickActions.register('h', () => quickActions.showHelp(), '显示帮助');

    // 初始化交易详情模态框
    window.tradeModal = new TradeDetailModal();
}

function exportCurrentTable() {
    const activeTable = document.querySelector('.data-table table:not([style*="display: none"])');
    if (activeTable) {
        const exporter = new TableExporter(activeTable);
        const timestamp = new Date().toISOString().slice(0, 10);
        exporter.exportCSV(`trades_${timestamp}.csv`);
    }
}

function focusFilterInput() {
    const filterInput = document.querySelector('[data-filter-target]');
    if (filterInput) {
        filterInput.focus();
    }
}

// 自动初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTableEnhancements);
} else {
    initTableEnhancements();
}

// 导出到全局
window.SortableTable = SortableTable;
window.FilterableTable = FilterableTable;
window.QuickActions = QuickActions;
window.TableExporter = TableExporter;
window.ResponsiveTable = ResponsiveTable;
window.TradeDetailModal = TradeDetailModal;
