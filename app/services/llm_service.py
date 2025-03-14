import json
import requests
from typing import Generator, Dict, List, Any, Union, Optional, Tuple
import threading
import time
import queue
from app.utils.logger import get_logger
from app.utils.api_utils import APIUtils

# Get logger
logger = get_logger()


class LLMService:
    """
    Service for interacting with Large Language Model APIs like OpenAI.
    Supports both streaming and non-streaming responses.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        Initialize the LLM service with API configuration.

        Args:
            api_url: Base URL for the API (e.g., 'https://api.openai.com'), will be append '/v1/chat/completions'
            api_key: API key for authentication
            model: Model name to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            timeout: Timeout in seconds for API requests
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = int(timeout or 60)

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Send a chat request to the LLM API.

        Args:
            messages: List of message objects with 'role' and 'content' keys
            stream: Whether to stream the response
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API

        Returns:
            For stream=False: String response from the LLM
            For stream=True: Generator yielding response chunks
        """
        if not self.api_url:
            error_msg = "API URL is not configured"
            logger.error(error_msg)
            return self._handle_error(error_msg, stream)

        if not self.api_key:
            error_msg = "API Key is not configured"
            logger.error(error_msg)
            return self._handle_error(error_msg, stream)

        # Standardize the API URL
        api_url = APIUtils.format_api_url(self.api_url)
        logger.debug(f"Standardized API URL: {api_url}")

        # Build request headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        # Add max_tokens if specified
        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Add any additional kwargs to the payload
        for key, value in kwargs.items():
            payload[key] = value

        if stream:
            return self._stream_chat(api_url, headers, payload)
        else:
            return self._non_stream_chat(api_url, headers, payload)

    def _stream_chat(
        self, api_url: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """
        Handle streaming chat responses from the API.

        Args:
            api_url: The API URL to call
            headers: Request headers
            payload: Request payload

        Returns:
            Generator yielding response chunks
        """
        try:
            logger.debug(f"Initiating streaming API request to: {api_url}")

            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                stream=True,
            )

            logger.debug(f"API streaming response status code: {response.status_code}")

            if response.status_code == 200:
                logger.info("Successfully received streaming API response")

                for line in response.iter_lines():
                    if not line:
                        continue

                    line = line.decode("utf-8")

                    # Skip keep-alive lines
                    if line.startswith(":"):
                        continue

                    # Remove "data: " prefix
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        # Handle stream end
                        if data == "[DONE]":
                            logger.debug("Stream completed")
                            break

                        try:
                            json_data = json.loads(data)
                            delta = json_data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                # Yield the content chunk
                                yield content

                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to parse JSON from chunk: {data}, Error: {str(e)}"
                            )

                        except Exception as e:
                            logger.error(f"Error processing stream chunk: {str(e)}")
                            yield json.dumps(
                                {"error": f"Error processing stream: {str(e)}"}
                            )

            else:
                # Handle error response
                error_msg = self._get_error_message(response)
                logger.error(f"API request failed: {error_msg}")
                yield json.dumps({"error": error_msg})

        except requests.exceptions.Timeout:
            error_msg = f"API request timed out after {self.timeout} seconds"
            logger.error(error_msg)
            yield json.dumps({"error": error_msg})

        except requests.exceptions.RequestException as e:
            error_msg = f"API request error: {str(e)}"
            logger.error(error_msg)
            yield json.dumps({"error": error_msg})

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            yield json.dumps({"error": error_msg})

    def _non_stream_chat(
        self, api_url: str, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> str:
        """
        Handle non-streaming chat responses from the API.

        Args:
            api_url: The API URL to call
            headers: Request headers
            payload: Request payload

        Returns:
            String response from the LLM
        """
        try:
            logger.debug(f"Initiating non-streaming API request to: {api_url}")

            response = requests.post(
                api_url, headers=headers, json=payload, timeout=self.timeout
            )

            logger.debug(
                f"API non-streaming response status code: {response.status_code}"
            )

            if response.status_code == 200:
                api_response = response.json()
                content = api_response["choices"][0]["message"]["content"]
                logger.info(
                    f"Successfully received API response, length: {len(content)}"
                )
                logger.debug(f"Response content (first 100 chars): {content[:100]}...")
                return content
            else:
                error_msg = self._get_error_message(response)
                logger.error(f"API request failed: {error_msg}")
                return f"Error: {error_msg}"

        except requests.exceptions.Timeout:
            error_msg = f"API request timed out after {self.timeout} seconds"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        except requests.exceptions.RequestException as e:
            error_msg = f"API request error: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            return f"Error: {error_msg}"

    def _get_error_message(self, response: requests.Response) -> str:
        """Extract error message from API response"""
        try:
            error_response = response.json()
            # Try to get detailed error message from different API formats
            if "error" in error_response:
                if isinstance(error_response["error"], dict):
                    return error_response["error"].get(
                        "message", str(error_response["error"])
                    )
                return str(error_response["error"])
            return f"Status code: {response.status_code}, Response: {response.text}"
        except:
            return f"Status code: {response.status_code}, Response: {response.text[:200] if response.text else 'No response content'}"

    def _handle_error(
        self, error_msg: str, stream: bool
    ) -> Union[str, Generator[str, None, None]]:
        """Handle errors based on streaming mode"""
        if stream:

            def error_generator():
                yield json.dumps({"error": error_msg})

            return error_generator()
        else:
            return f"Error: {error_msg}"


class LLMServicePool:
    """
    A pool of LLM services that manages multiple service instances and
    handles queueing requests when all services are busy.
    """

    def __init__(
        self,
        service_configs: List[Dict[str, Any]],
        max_queue_size: int = 100,
        request_timeout: int = 300,
    ):
        """
        Initialize the LLM service pool with multiple service configurations.

        Args:
            service_configs: List of dictionaries containing configuration for each service
                Each dict should have: api_url, api_key, model, and timeout (optional)
            max_queue_size: Maximum number of requests that can be queued
            request_timeout: Maximum time (seconds) a request can wait in queue
        """
        self.services = []
        self.service_locks = []
        self.max_queue_size = max_queue_size
        self.request_timeout = request_timeout
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        # Track the last service index for round-robin selection
        self.last_service_index = -1
        # Lock to protect the last_service_index variable
        self.selection_lock = threading.Lock()
        # Flag for pool shutdown
        self.is_shutdown = False

        # Initialize services and locks
        for config in service_configs:
            service = LLMService(
                api_url=config.get("api_url"),
                api_key=config.get("api_key"),
                model=config.get("model"),
                timeout=config.get("timeout"),
            )
            self.services.append(service)
            self.service_locks.append(threading.RLock())

        if not self.services:
            logger.warning("LLM Service Pool initialized with no services!")

        logger.info(f"LLM Service Pool initialized with {len(self.services)} services")

        # Start queue processing thread
        self.queue_processor = threading.Thread(target=self._process_queue, daemon=True)
        self.queue_processor.start()

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        wait: bool = True,
        **kwargs,
    ) -> Union[str, Generator[str, None, None], None]:
        """
        Send a chat request to an available LLM service in the pool.
        If all services are busy, either queue the request or return None based on wait parameter.

        Args:
            messages: List of message objects with 'role' and 'content' keys
            stream: Whether to stream the response
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            wait: If True, wait for an available service; if False, return None if no service is available
            **kwargs: Additional parameters to pass to the API

        Returns:
            For stream=False: String response from the LLM, or None if no service available and wait=False
            For stream=True: Generator yielding response chunks, or None if no service available and wait=False
        """
        # Try to get an available service immediately
        service_index, service = self._get_available_service()

        if service is not None:
            # Service available, use it directly
            try:
                logger.debug(f"Using service {service_index} for immediate request")
                return self._execute_chat_with_service(
                    service_index,
                    service,
                    messages,
                    stream,
                    temperature,
                    max_tokens,
                    **kwargs,
                )
            finally:
                # Release the service lock
                self.service_locks[service_index].release()
                logger.debug(f"Released service {service_index}")

        elif wait:
            # All services busy, queue the request if waiting is allowed
            logger.debug("All services busy, queueing request")
            return self._queue_request(
                messages, stream, temperature, max_tokens, **kwargs
            )

        else:
            # All services busy and not waiting, return None
            logger.info("All services busy and wait=False, returning None")
            return None

    def _get_available_service(self) -> Tuple[Optional[int], Optional[LLMService]]:
        """
        Try to acquire an available service from the pool using round-robin.

        Returns:
            Tuple of (service_index, service) or (None, None) if none available
        """
        if not self.services:
            logger.warning("No services in the pool")
            return None, None

        with self.selection_lock:
            # Get the number of services
            num_services = len(self.services)
            # Try each service, starting from the next one in round-robin order
            for i in range(num_services):
                # Calculate the next index in a round-robin fashion
                index = (self.last_service_index + 1 + i) % num_services
                if self.service_locks[index].acquire(blocking=False):
                    logger.debug(f"Acquired service {index} (round-robin)")
                    # Update the last service index for next selection
                    self.last_service_index = index
                    return index, self.services[index]

        # None available
        return None, None

    def _queue_request(
        self,
        messages: List[Dict[str, str]],
        stream: bool,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Queue a request to be processed when a service becomes available.

        Returns:
            For stream=False: String response from the LLM
            For stream=True: Generator yielding response chunks
        """
        # For streaming requests, we need to create a queue to receive chunks
        result_queue = None
        if stream:
            result_queue = queue.Queue()

        # Create a result event for non-streaming requests
        result_event = threading.Event()
        result_container = {"response": None, "error": None}

        # Create request object
        request = {
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
            "result_event": result_event,
            "result_container": result_container,
            "result_queue": result_queue,
            "timestamp": time.time(),
        }

        try:
            # Add to queue, with timeout to prevent blocking forever if queue is full
            self.request_queue.put(request, timeout=2)
            logger.debug("Request added to queue")

            if stream:
                # For streaming, return a generator that yields from the result queue
                def stream_generator():
                    while True:
                        try:
                            chunk = result_queue.get(timeout=self.request_timeout)
                            if chunk is None:  # None is our sentinel value
                                break
                            yield chunk
                        except queue.Empty:
                            logger.warning("Timeout while waiting for stream chunks")
                            yield json.dumps({"error": "Timeout waiting for response"})
                            break

                return stream_generator()
            else:
                # For non-streaming, wait on the event
                if not result_event.wait(timeout=self.request_timeout):
                    logger.warning("Timeout while waiting for request to be processed")
                    return "Error: Request timed out waiting for available service"

                # Check if there was an error
                if result_container["error"]:
                    return f"Error: {result_container['error']}"

                return result_container["response"]

        except queue.Full:
            error_msg = "Request queue is full, try again later"
            logger.warning(error_msg)

            if stream:

                def error_stream():
                    yield json.dumps({"error": error_msg})

                return error_stream()
            else:
                return f"Error: {error_msg}"

    def _process_queue(self):
        """
        Background thread that processes queued requests when services become available.
        """
        logger.info("Queue processor thread started")

        while not self.is_shutdown:
            try:
                # Get next request from queue
                try:
                    request = self.request_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Check if request is too old
                if time.time() - request["timestamp"] > self.request_timeout:
                    logger.warning("Discarding expired request from queue")

                    # Notify waiting threads
                    if not request["stream"]:
                        request["result_container"][
                            "error"
                        ] = "Request expired in queue"
                        request["result_event"].set()
                    else:
                        request["result_queue"].put(
                            json.dumps({"error": "Request expired in queue"})
                        )
                        request["result_queue"].put(None)  # Signal end

                    # Mark as done
                    self.request_queue.task_done()
                    continue

                # Wait for an available service using round-robin
                service_acquired = False
                service_index = None

                while not service_acquired and not self.is_shutdown:
                    # Try to get any available service using round-robin
                    service_index, service = self._get_available_service()

                    if service_index is not None:
                        service_acquired = True
                    else:
                        # No service available, sleep briefly before trying again
                        time.sleep(0.1)

                if self.is_shutdown:
                    break

                # Get service and params from request
                service = self.services[service_index]
                messages = request["messages"]
                stream = request["stream"]
                temperature = request["temperature"]
                max_tokens = request["max_tokens"]
                kwargs = request["kwargs"]

                logger.debug(f"Processing queued request with service {service_index}")

                try:
                    if stream:
                        # For streaming, we need to forward chunks to the result queue
                        chunks_generator = service.chat(
                            messages=messages,
                            stream=True,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            **kwargs,
                        )

                        # Forward all chunks to the result queue
                        for chunk in chunks_generator:
                            request["result_queue"].put(chunk)

                        # Signal end of stream
                        request["result_queue"].put(None)
                    else:
                        # For non-streaming, get the response and set it in the container
                        response = service.chat(
                            messages=messages,
                            stream=False,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            **kwargs,
                        )

                        request["result_container"]["response"] = response
                        request["result_event"].set()

                except Exception as e:
                    logger.error(f"Error processing queued request: {str(e)}")

                    if stream:
                        request["result_queue"].put(json.dumps({"error": str(e)}))
                        request["result_queue"].put(None)  # Signal end
                    else:
                        request["result_container"]["error"] = str(e)
                        request["result_event"].set()

                finally:
                    # Release the service lock
                    self.service_locks[service_index].release()
                    logger.debug(f"Released service {service_index}")

                    # Mark task as done
                    self.request_queue.task_done()

            except Exception as e:
                logger.error(f"Error in queue processor: {str(e)}")
                logger.exception(e)
                # Continue processing to maintain thread

    def _execute_chat_with_service(
        self,
        service_index: int,
        service: LLMService,
        messages: List[Dict[str, str]],
        stream: bool,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Execute a chat request with the specified service.
        The service lock should be acquired before calling this method.
        """
        logger.debug(f"Executing request with service {service_index}")

        try:
            return service.chat(
                messages=messages,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as e:
            error_msg = f"Error in service {service_index}: {str(e)}"
            logger.error(error_msg)

            if stream:

                def error_stream():
                    yield json.dumps({"error": error_msg})

                return error_stream()
            else:
                return f"Error: {error_msg}"

    def shutdown(self):
        """
        Shutdown the service pool cleanly, stopping the queue processor.
        """
        logger.info("Shutting down LLM Service Pool")
        self.is_shutdown = True

        # Wait for queue processor to terminate
        if self.queue_processor and self.queue_processor.is_alive():
            self.queue_processor.join(timeout=5)

        # Process remaining items in the queue
        while not self.request_queue.empty():
            try:
                request = self.request_queue.get_nowait()
                logger.debug("Cancelling queued request during shutdown")

                if request["stream"]:
                    request["result_queue"].put(
                        json.dumps({"error": "Service pool shutting down"})
                    )
                    request["result_queue"].put(None)
                else:
                    request["result_container"]["error"] = "Service pool shutting down"
                    request["result_event"].set()

                self.request_queue.task_done()
            except queue.Empty:
                break
