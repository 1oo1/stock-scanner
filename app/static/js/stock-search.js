/**
 * Stock Search Component
 * 
 * This component creates a search input with dropdown results that
 * can be used throughout the application.
 * Uses Tailwind CSS for styling.
 */

class StockSearchComponent {
  constructor(containerElement, options = {}) {
    this.container = typeof containerElement === 'string'
      ? document.querySelector(containerElement)
      : containerElement;

    if (!this.container) {
      throw new Error('Container element not found');
    }

    // Default options
    this.options = {
      placeholder: '搜索代码或名称...',
      maxResults: 10,
      debounceTime: 300,
      onSelect: null,
      ...options
    };

    this.searchResults = [];
    this.selectedIndex = -1;
    this.selectedItem = null;

    this._initializeDOM();
    this._setupEventListeners();
  }

  _initializeDOM() {
    // Create component markup with Tailwind CSS classes
    this.container.classList.add('relative', 'w-full', 'max-w-lg', 'mx-auto');
    this.container.innerHTML = `
      <div class="relative">
        <input type="text" class="stock-search-input w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-colors" placeholder="${this.options.placeholder}">
        <div class="stock-search-spinner absolute right-3 top-1/2 transform -translate-y-1/2 hidden" role="status">
          <i data-lucide="loader-2" class="h-5 w-5 text-blue-500 animate-spin"></i>
          <span class="sr-only">Loading...</span>
        </div>
      </div>
      <div class="stock-search-dropdown absolute w-full mt-1 bg-white rounded-md shadow-lg overflow-hidden z-50 hidden">
        <ul class="stock-search-results max-h-60 overflow-y-auto py-1"></ul>
      </div>
    `;

    lucide.createIcons();
  }

  _setupEventListeners() {
    // Cache DOM elements
    this.searchInput = this.container.querySelector('.stock-search-input');
    this.resultsDropdown = this.container.querySelector('.stock-search-dropdown');
    this.resultsList = this.container.querySelector('.stock-search-results');
    this.spinner = this.container.querySelector('.stock-search-spinner');

    // Debounced search function
    this.debouncedSearch = debounce((query) => {
      if (query.length < 2) {
        this.hideResults();
        return;
      }
      this._performSearch(query);
    }, this.options.debounceTime);

    // Input event for search
    this.searchInput.addEventListener('input', (e) => {
      const query = e.target.value.trim();
      this.debouncedSearch(query);
    });

    // Handle keyboard navigation
    this.searchInput.addEventListener('keydown', this._handleKeyDown.bind(this));

    // Handle result selection via event delegation
    this.resultsList.addEventListener('click', (e) => {
      const listItem = e.target.closest('li');
      if (listItem) {
        const index = parseInt(listItem.dataset.index, 10);
        if (!isNaN(index) && index >= 0 && index < this.searchResults.length) {
          this._handleResultSelect(this.searchResults[index]);
        }
      }
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
      if (!this.container.contains(e.target)) {
        this.hideResults();
      }
    });
  }

  _handleKeyDown(e) {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        this._selectNextResult();
        break;
      case 'ArrowUp':
        e.preventDefault();
        this._selectPreviousResult();
        break;
      case 'Enter':
        e.preventDefault();
        this._selectCurrentResult();
        break;
      case 'Escape':
        e.preventDefault();
        this.hideResults();
        break;
    }
  }

  async _performSearch(query) {
    try {
      this.spinner.classList.remove('hidden');

      // Call the search API
      const data = await apiService.get('/search', { q: query, limit: this.options.maxResults });

      this.selectedItem = null;
      this.searchResults = data.results || [];
      this._renderResults();
    } catch (error) {
      console.error('Search failed:', error);
      this.searchResults = [];
      this._renderError('搜索出错，请稍后再试');
    } finally {
      this.spinner.classList.add('hidden');
    }
  }

  _renderResults() {
    this.resultsList.innerHTML = '';

    if (this.searchResults.length === 0) {
      this._renderError('没有找到匹配的股票');
      return;
    }

    this.searchResults.forEach((stock, index) => {
      const listItem = document.createElement('li');
      listItem.className = 'px-4 py-2 hover:bg-gray-100 cursor-pointer';
      listItem.dataset.index = index;
      listItem.innerHTML = `
        <div class="flex justify-between items-center">
          <div>
            <span class="font-medium">${stock.stock_code}</span> - ${stock.stock_name}
          </div>
          <span class="bg-gray-100 text-gray-800 text-xs font-medium px-2 py-0.5 rounded">${stock.market_type}</span>
        </div>
      `;
      this.resultsList.appendChild(listItem);
    });

    this.showResults();
  }

  _renderError(message) {
    this.resultsList.innerHTML = `
      <li class="px-4 py-3 text-center text-gray-500">
        ${message}
      </li>
    `;
    this.showResults();
  }

  _handleResultSelect(stockData) {
    // Clear the input and hide dropdown
    this.searchInput.value = `${stockData.market_type} - ${stockData.stock_code} - ${stockData.stock_name}`;
    this.hideResults();

    this.selectedItem = stockData;
    // Call callback if provided
    if (typeof this.options.onSelect === 'function') {
      this.options.onSelect(stockData);
    }
  }

  _selectNextResult() {
    if (this.searchResults.length === 0) return;

    this.selectedIndex = Math.min(this.selectedIndex + 1, this.searchResults.length - 1);
    this._highlightSelectedResult();
  }

  _selectPreviousResult() {
    if (this.searchResults.length === 0) return;

    this.selectedIndex = Math.max(this.selectedIndex - 1, 0);
    this._highlightSelectedResult();
  }

  _selectCurrentResult() {
    if (this.selectedIndex >= 0 && this.selectedIndex < this.searchResults.length) {
      const selectedStock = this.searchResults[this.selectedIndex];
      this._handleResultSelect(selectedStock);
    }
  }

  _highlightSelectedResult() {
    const items = this.resultsList.querySelectorAll('li');
    items.forEach((item, index) => {
      if (index === this.selectedIndex) {
        item.classList.add('bg-gray-100');
      } else {
        item.classList.remove('bg-gray-100');
      }
    });
  }

  showResults() {
    this.resultsDropdown.classList.remove('hidden');
  }

  hideResults() {
    this.resultsDropdown.classList.add('hidden');
    this.selectedIndex = -1;
  }

  // Public methods to control the component
  setValue(value) {
    this.searchInput.value = value;
  }

  getValue() {
    return this.searchInput.value;
  }

  focus() {
    this.searchInput.focus();
  }

  clear() {
    this.searchInput.value = '';
    this.hideResults();
  }
}

// Initialize all stock search components when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
  // Auto-initialize any elements with data-stock-search attribute
  const searchContainers = document.querySelectorAll('[data-stock-search]');

  searchContainers.forEach(container => {
    // Parse options from data attributes
    const options = {};

    if (container.dataset.placeholder) {
      options.placeholder = container.dataset.placeholder;
    }

    if (container.dataset.maxResults) {
      options.maxResults = parseInt(container.dataset.maxResults, 10);
    }

    if (container.dataset.debounceTime) {
      options.debounceTime = parseInt(container.dataset.debounceTime, 500);
    }

    // Check if onSelect is a valid function
    if (container.dataset.onSelect) {
      const onSelectFunc = window[container.dataset.onSelect];
      if (typeof onSelectFunc === 'function') {
        options.onSelect = onSelectFunc;
      }
    }

    // Initialize the component
    const searchComponent = new StockSearchComponent(container, options);

    // Store the component instance on the element
    container.stockSearch = searchComponent;
  });
});

// Make it available globally
window.StockSearchComponent = StockSearchComponent;
