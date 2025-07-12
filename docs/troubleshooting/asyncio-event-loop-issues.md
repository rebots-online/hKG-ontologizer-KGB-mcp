# AsyncIO Event Loop Issues in Gradio Applications

## Problem Summary

When building Gradio applications that use async functions internally, you may encounter the error:
```
"Error in async execution: asyncio.run() cannot be called from a running event loop"
```

This occurs when `asyncio.run()` is called from within an already running event loop, which is common in Gradio applications since they run in their own event loop.

## Root Cause Analysis

### Why This Happens
1. **Gradio's Event Loop**: Gradio applications run within an existing asyncio event loop
2. **Nested Event Loops**: `asyncio.run()` creates a new event loop and cannot be called from within an existing one
3. **Async Function Wrappers**: Synchronous wrapper functions attempting to use `asyncio.run()` fail when called from Gradio's event loop

### Code Patterns That Cause Issues

**❌ Problematic Pattern:**
```python
def sync_wrapper(input_data):
    """This will fail in Gradio"""
    import asyncio
    try:
        result = asyncio.run(async_function(input_data))  # ❌ Fails in running loop
        return result
    except Exception as e:
        return {"error": f"Error in async execution: {str(e)}"}
```

**❌ Another Problematic Pattern:**
```python
async def async_wrapper(input_data):
    """This also fails"""
    import asyncio
    result = asyncio.run(another_async_function(input_data))  # ❌ Still wrong
    return result
```

## Solutions

### Solution 1: Proper Async/Await Pattern
**✅ Correct Approach:**
```python
async def async_wrapper(input_data):
    """Use await instead of asyncio.run()"""
    result = await async_function(input_data)  # ✅ Correct
    return result
```

### Solution 2: Thread Pool for Event Loop Isolation
**✅ Robust Synchronous Wrapper:**
```python
def sync_wrapper_for_gradio(input_data):
    """Handles event loop detection and isolation"""
    import asyncio
    import concurrent.futures
    
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in a running loop - use thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(async_function(input_data))
                )
                return future.result()
        else:
            # Safe to use run_until_complete
            return loop.run_until_complete(async_function(input_data))
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(async_function(input_data))
```

### Solution 3: Nest AsyncIO (Alternative)
**✅ Using nest_asyncio:**
```python
def sync_wrapper_with_nest_asyncio(input_data):
    """Enable nested event loops"""
    import asyncio
    import nest_asyncio
    
    # Enable nested event loops
    nest_asyncio.apply()
    
    try:
        result = asyncio.run(async_function(input_data))
        return result
    except Exception as e:
        return {"error": f"Error: {str(e)}"}
```

## Implementation Details

### File: `app.py` Lines 1796-1831
The fix implemented in this project uses the Thread Pool approach:

```python
def process_with_config_sync(text_input, uploaded_file, provider, model, api_url, api_key):
    """Synchronous wrapper for Gradio that properly handles async calls."""
    import asyncio
    import nest_asyncio
    
    # Enable nested event loops (important for Gradio)
    try:
        nest_asyncio.apply()
    except:
        pass  # nest_asyncio might not be available
    
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in a running loop, we need to use create_task
            import concurrent.futures
            
            # Create a new event loop in a thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(
                        process_with_config_async(text_input, uploaded_file, provider, model, api_url, api_key)
                    )
                )
                return future.result()
        else:
            # We can safely use run_until_complete
            return loop.run_until_complete(
                process_with_config_async(text_input, uploaded_file, provider, model, api_url, api_key)
            )
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(
            process_with_config_async(text_input, uploaded_file, provider, model, api_url, api_key)
        )
```

## Key Changes Made

1. **Removed `asyncio.run()` from async functions** - Lines 1371, 1785
2. **Replaced with proper `await` calls** in async contexts
3. **Created thread-pool-based synchronous wrapper** for Gradio compatibility
4. **Updated Gradio event handlers** to use the synchronous wrapper

## Testing and Validation

After implementing the fix:
- ✅ No more "cannot be called from a running event loop" errors
- ✅ Gradio interface functions correctly
- ✅ Async functions execute properly within the application
- ✅ Real-time progress updates work as expected

## Prevention Guidelines

1. **Never use `asyncio.run()` inside async functions**
2. **Use `await` for calling async functions from async contexts**
3. **For Gradio integration, use thread pool isolation for sync wrappers**
4. **Test with both standalone and Gradio-hosted execution**
5. **Consider using `nest_asyncio` for simpler cases**

## Dependencies for Solutions

```txt
# For thread pool approach (built-in)
concurrent.futures  # Standard library

# For nested event loops approach
nest-asyncio>=1.5.0
```

## Related Issues

- Gradio applications with MCP server integration
- FastAPI + Gradio hybrid applications
- Jupyter notebook environments with existing event loops
- Any application mixing sync and async patterns

## References

- [Python AsyncIO Documentation](https://docs.python.org/3/library/asyncio.html)
- [Gradio AsyncIO Compatibility](https://gradio.app/docs/)
- [nest_asyncio Package](https://pypi.org/project/nest-asyncio/)