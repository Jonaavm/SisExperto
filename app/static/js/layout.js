// Layout JavaScript - Sistema Experto

document.addEventListener('DOMContentLoaded', function() {
    // Clean corrupted localStorage data on page load
    try {
        const storageManager = window.StorageManager || (window.StorageManager = {
            cleanCorruptedData: function() {
                try {
                    const keys = Object.keys(localStorage);
                    let cleaned = 0;
                    
                    keys.forEach(key => {
                        try {
                            const item = localStorage.getItem(key);
                            if (item) {
                                JSON.parse(item); // Test if it's valid JSON
                            }
                        } catch (error) {
                            console.warn(`Removing corrupted localStorage item: ${key}`);
                            localStorage.removeItem(key);
                            cleaned++;
                        }
                    });
                    
                    if (cleaned > 0) {
                        console.log(`Cleaned ${cleaned} corrupted localStorage items`);
                    }
                    
                    return cleaned;
                } catch (error) {
                    console.error('Error cleaning localStorage:', error);
                    return 0;
                }
            }
        });
        
        storageManager.cleanCorruptedData();
    } catch (error) {
        console.error('Error initializing storage cleanup:', error);
    }
    
    // Mobile navigation toggle
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', function() {
            navToggle.classList.toggle('active');
            navLinks.classList.toggle('active');
        });
        
        // Close mobile nav when clicking on a link
        const navLinkElements = document.querySelectorAll('.nav-link');
        navLinkElements.forEach(link => {
            link.addEventListener('click', function() {
                navToggle.classList.remove('active');
                navLinks.classList.remove('active');
            });
        });
        
        // Close mobile nav when clicking outside
        document.addEventListener('click', function(event) {
            if (!navToggle.contains(event.target) && !navLinks.contains(event.target)) {
                navToggle.classList.remove('active');
                navLinks.classList.remove('active');
            }
        });
    }
    
    // Highlight active navigation link
    highlightActiveNavLink();
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.opacity = '0';
            setTimeout(() => {
                alert.remove();
            }, 300);
        }, 5000);
    });
    
    // Loading state management
    window.showLoading = function(element) {
        if (element) {
            element.disabled = true;
            const originalText = element.innerHTML;
            element.innerHTML = '<span class="loading"></span> Cargando...';
            element.dataset.originalText = originalText;
        }
    };
    
    window.hideLoading = function(element) {
        if (element && element.dataset.originalText) {
            element.disabled = false;
            element.innerHTML = element.dataset.originalText;
            delete element.dataset.originalText;
        }
    };
    
    // Global error handler
    window.addEventListener('error', function(event) {
        console.error('Global error:', event.error);
        showMessage('Ha ocurrido un error inesperado. Por favor, recarga la página.', 'error');
    });
    
    // Global message system
    window.showMessage = function(message, type = 'info', duration = 5000) {
        // Remove existing messages
        const existingMessages = document.querySelectorAll('.global-message');
        existingMessages.forEach(msg => msg.remove());
        
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `alert alert-${type} global-message`;
        messageDiv.textContent = message;
        messageDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
            max-width: 500px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            animation: slideInRight 0.3s ease;
        `;
        
        // Add close button
        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '×';
        closeBtn.style.cssText = `
            background: none;
            border: none;
            font-size: 20px;
            font-weight: bold;
            float: right;
            cursor: pointer;
            margin-left: 10px;
            opacity: 0.7;
        `;
        closeBtn.addEventListener('click', () => messageDiv.remove());
        messageDiv.appendChild(closeBtn);
        
        document.body.appendChild(messageDiv);
        
        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (messageDiv.parentNode) {
                    messageDiv.style.animation = 'slideOutRight 0.3s ease';
                    setTimeout(() => messageDiv.remove(), 300);
                }
            }, duration);
        }
    };
    
    // Form validation helper
    window.validateForm = function(formElement) {
        if (!formElement) return false;
        
        const requiredFields = formElement.querySelectorAll('[required]');
        let isValid = true;
        
        requiredFields.forEach(field => {
            const value = field.value.trim();
            const fieldContainer = field.closest('.form-group') || field.parentElement;
            
            // Remove existing error styling
            field.classList.remove('error');
            const existingError = fieldContainer.querySelector('.error-message');
            if (existingError) existingError.remove();
            
            if (!value) {
                isValid = false;
                field.classList.add('error');
                
                // Add error message
                const errorMsg = document.createElement('div');
                errorMsg.className = 'error-message';
                errorMsg.textContent = 'Este campo es requerido';
                errorMsg.style.cssText = 'color: #dc3545; font-size: 0.8em; margin-top: 5px;';
                fieldContainer.appendChild(errorMsg);
            }
        });
        
        return isValid;
    };
    
    // Local storage helpers
    window.storageHelper = {
        set: function(key, value) {
            try {
                // Validate that the value can be serialized
                const serialized = JSON.stringify(value);
                localStorage.setItem(key, serialized);
                return true;
            } catch (error) {
                console.error('Error saving to localStorage:', error);
                return false;
            }
        },
        
        get: function(key, defaultValue = null) {
            try {
                const item = localStorage.getItem(key);
                if (!item) return defaultValue;
                
                // Try to parse the JSON
                try {
                    return JSON.parse(item);
                } catch (parseError) {
                    console.warn(`Invalid JSON in localStorage for key "${key}":`, item);
                    // Remove corrupted data
                    localStorage.removeItem(key);
                    return defaultValue;
                }
            } catch (error) {
                console.error('Error reading from localStorage:', error);
                return defaultValue;
            }
        },
        
        remove: function(key) {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (error) {
                console.error('Error removing from localStorage:', error);
                return false;
            }
        },
        
        clear: function() {
            try {
                localStorage.clear();
                return true;
            } catch (error) {
                console.error('Error clearing localStorage:', error);
                return false;
            }
        },
        
        // Clean corrupted data from localStorage
        cleanCorruptedData: function() {
            try {
                const keys = Object.keys(localStorage);
                let cleaned = 0;
                
                keys.forEach(key => {
                    try {
                        const item = localStorage.getItem(key);
                        if (item) {
                            JSON.parse(item); // Test if it's valid JSON
                        }
                    } catch (error) {
                        console.warn(`Removing corrupted localStorage item: ${key}`);
                        localStorage.removeItem(key);
                        cleaned++;
                    }
                });
                
                if (cleaned > 0) {
                    console.log(`Cleaned ${cleaned} corrupted localStorage items`);
                }
                
                return cleaned;
            } catch (error) {
                console.error('Error cleaning localStorage:', error);
                return 0;
            }
        }
    };
    
    // Performance monitoring
    if ('performance' in window) {
        window.addEventListener('load', function() {
            setTimeout(function() {
                const perfData = performance.timing;
                const loadTime = perfData.loadEventEnd - perfData.navigationStart;
                console.log(`Page load time: ${loadTime}ms`);
                
                // Log slow pages
                if (loadTime > 3000) {
                    console.warn('Slow page load detected');
                }
            }, 0);
        });
    }
});

// Highlight current page in navigation
function highlightActiveNavLink() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        
        // Check if link href matches current path
        const linkPath = new URL(link.href).pathname;
        if (linkPath === currentPath || 
            (currentPath === '/' && linkPath === '/') ||
            (currentPath !== '/' && linkPath !== '/' && currentPath.startsWith(linkPath))) {
            link.classList.add('active');
        }
    });
}

// Utility functions for API calls
window.apiHelper = {
    async request(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        const config = {
            ...defaultOptions,
            ...options,
            headers: {
                ...defaultOptions.headers,
                ...options.headers,
            },
        };
        
        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                try {
                    return await response.json();
                } catch (jsonError) {
                    console.error('Error parsing JSON response:', jsonError);
                    const textResponse = await response.text();
                    console.warn('Response was not valid JSON:', textResponse);
                    return { error: 'Invalid JSON response', raw: textResponse };
                }
            } else {
                return await response.text();
            }
        } catch (error) {
            console.error('API request failed:', error);
            showMessage('Error de conexión. Por favor, verifica tu conexión a internet.', 'error');
            throw error;
        }
    },
    
    async get(url, options = {}) {
        return this.request(url, { ...options, method: 'GET' });
    },
    
    async post(url, data, options = {}) {
        return this.request(url, {
            ...options,
            method: 'POST',
            body: JSON.stringify(data),
        });
    },
    
    async put(url, data, options = {}) {
        return this.request(url, {
            ...options,
            method: 'PUT',
            body: JSON.stringify(data),
        });
    },
    
    async delete(url, options = {}) {
        return this.request(url, { ...options, method: 'DELETE' });
    }
};

// File upload helper
window.fileHelper = {
    validateFile(file, allowedTypes = ['.csv'], maxSize = 10 * 1024 * 1024) {
        const errors = [];
        
        if (!file) {
            errors.push('No se ha seleccionado ningún archivo');
            return errors;
        }
        
        // Check file type
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        if (allowedTypes.length > 0 && !allowedTypes.includes(fileExtension)) {
            errors.push(`Tipo de archivo no permitido. Tipos permitidos: ${allowedTypes.join(', ')}`);
        }
        
        // Check file size
        if (file.size > maxSize) {
            const maxSizeMB = (maxSize / (1024 * 1024)).toFixed(1);
            errors.push(`El archivo es demasiado grande. Tamaño máximo: ${maxSizeMB}MB`);
        }
        
        return errors;
    },
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
};

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .error {
        border-color: #dc3545 !important;
        box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25) !important;
    }
`;
document.head.appendChild(style);

// Global error handling for unhandled promise rejections and JSON errors
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    
    // Check if it's a JSON parsing error
    if (event.reason && event.reason.message && event.reason.message.includes('JSON')) {
        console.warn('JSON parsing error detected, cleaning localStorage...');
        try {
            // Clean potentially corrupted localStorage
            if (window.StorageManager && window.StorageManager.cleanCorruptedData) {
                window.StorageManager.cleanCorruptedData();
            }
        } catch (cleanupError) {
            console.error('Error during localStorage cleanup:', cleanupError);
        }
        
        // Prevent the error from showing in console if it's just a JSON parsing issue
        event.preventDefault();
    }
});

// Global error handler for synchronous errors
window.addEventListener('error', function(event) {
    // Log the error but don't show it to users unless it's critical
    if (event.error && event.error.message && 
        (event.error.message.includes('JSON') || event.error.message.includes('localStorage'))) {
        console.warn('Storage-related error suppressed:', event.error.message);
        event.preventDefault();
    }
});

console.log('Layout JavaScript loaded successfully with enhanced error handling');
