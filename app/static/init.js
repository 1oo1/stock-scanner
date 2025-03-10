// 初始化Lucide图标
lucide.createIcons();

// 创建防抖函数
function debounce(func, wait) {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => {
      func.apply(this, args);
    }, wait);
  };
}

// get cookie value by name
function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
}

// 全局API服务
const apiService = {
  async _fetch(url, data = {}, method = 'POST') {
    try {
      const response = await fetch(url, {
        method: method,
        headers: {
          'Content-Type': 'application/json',
          'X-CSRF-TOKEN': getCookie('csrf_access_token') || '',
        },
        body: JSON.stringify(data),
        credentials: 'include',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.error || 'unknown error'}`);
      }
      return response;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  },

  async post(url, data = {}) {
    const response = await this._fetch(url, data);
    return response.json();
  },

  /**
   * Returns a generator that yields JSON chunks as they arrive
   * Usage example:
   * ```javascript
   *   for await (const chunk of apiService.postStream('/api/data', { query: 'example' })) {
   *     console.log(chunk);
   *   }
   * ```
   */
  async *stream(url, data = {}) {
    const response = await this._fetch(url, data);
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        const text = decoder.decode(value, { stream: true });
        buffer += text;

        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the last partial line

        for (const line of lines) {
          if (line.trim()) {
            try {
              const json = JSON.parse(line);
              yield json;
            } catch (error) {
              console.error('Error parsing JSON:', error);
            }
          }
        }
      }

      // Handle any remaining buffer
      if (buffer.trim()) {
        try {
          const json = JSON.parse(buffer);
          yield json;
        } catch (error) {
          console.error('Error parsing JSON:', error);
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
};

window.apiService = apiService;