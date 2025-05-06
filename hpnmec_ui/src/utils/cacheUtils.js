/**
 * Simple in-memory cache implementation for API responses.
 * Can safely cache any serializable JS object, including those matching TypeScript interfaces.
 */

/**
 * @template T
 * @typedef {T} CachedItem
 * @description
 * For documentation: T can be any object, including those matching TypeScript interfaces
 * such as IUser, IHealthRecord, Appointment, etc.
 */

const cache = new Map();

/**
 * Default time-to-live for cache entries (in milliseconds)
 * @type {number}
 */
const DEFAULT_TTL = 5 * 60 * 1000; // 5 minutes

/**
 * Gets an item from the cache.
 * The returned object will retain its structure as when cached.
 * 
 * @template T
 * @param {string} key - The cache key
 * @returns {CachedItem<T>|null} The cached value or null if not found/expired
 */
export const getCacheItem = (key) => {
  if (!cache.has(key)) return null;
  
  const { value, expiry } = cache.get(key);
  
  // Check if the item has expired
  if (expiry && expiry < Date.now()) {
    cache.delete(key);
    return null;
  }
  
  return value;
};

/**
 * Sets an item in the cache with optional TTL.
 * 
 * @template T
 * @param {string} key - The cache key
 * @param {CachedItem<T>} value - The value to cache
 * @param {number} [ttl=DEFAULT_TTL] - Time-to-live in milliseconds
 */
export const setCacheItem = (key, value, ttl = DEFAULT_TTL) => {
  const expiry = ttl ? Date.now() + ttl : null;
  cache.set(key, { value, expiry });
};

/**
 * Removes an item from the cache.
 * 
 * @param {string} key - The cache key
 */
export const removeCacheItem = (key) => {
  cache.delete(key);
};

/**
 * Invalidates a cache item and any related items.
 * 
 * @param {string} keyPattern - The key or key pattern to invalidate
 */
export const invalidateCacheItem = (keyPattern) => {
  for (const key of cache.keys()) {
    if (key === keyPattern || key.startsWith(keyPattern)) {
      cache.delete(key);
    }
  }
};

/**
 * Clears the entire cache
 */
export const clearCache = () => {
  cache.clear();
};

/**
 * Creates a consistent cache key, optionally incorporating query parameters.
 * 
 * @param {string} baseKey - The base key
 * @param {Object} [params] - Optional query parameters
 * @returns {string} The generated cache key
 */
export const createCacheKey = (baseKey, params = null) => {
  if (!params) return baseKey;
  
  // Sort keys to ensure consistent order
  const sortedKeys = Object.keys(params).sort();
  const paramsString = sortedKeys
    .filter(key => params[key] !== undefined && params[key] !== null)
    .map(key => `${key}=${params[key]}`)
    .join('&');
    
  return paramsString ? `${baseKey}?${paramsString}` : baseKey;
};
