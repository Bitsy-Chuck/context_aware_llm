import json
import logging
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from queue import Queue, Empty
from typing import Dict, Set, Tuple, Optional, Any
from urllib.parse import unquote, urlparse, urlunparse

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DocCrawler:
    def __init__(self,
                 output_dir: str = "docs",
                 num_workers: int = 10,
                 max_retries: int = 3,
                 save_interval: int = 300,
                 base_pattern: str = r"https://www\.tradingview\.com/charting-library-docs/latest"):

        # Basic configuration
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.max_retries = max_retries
        self.save_interval = save_interval
        self.base_pattern = base_pattern

        # State tracking
        self.processed_urls: Set[str] = set()
        self.url_mapping: Dict[str, int] = {}
        self.current_id: int = 1
        self.active = True
        self.pending_tasks = 0

        # Bearer tokens for API access
        self.bearer_tokens = [
            "jina_e85c94cc3ec645cd8b9746c69a110f39zLteW6P_eyqLMrxY90yoETowTjv-",
            "jina_b8fe6ec49d8b4a719276411d5f5c58a4gruCVSd-1zpFfM_tpKFD8NLs9nvC",
            "jina_997d983d01fd48ac93b8f5978b19b4dcdqyNZktrSEtzl2oyi1raQeteBKkl",
            "jina_8bc4257ee8f44f3cbce2b343074259beSAVKSkFZMPxg0a2QqSh6htx5-nYJ",
            "jina_6ebccf7b64564388af56a8eecf43a2e6ez6xb2NGR1yVzsNdaaH6UpYDXebL",
            "jina_788e462ef048444491c7d755840448889ySq-2H5EQvgaS2whyc226UevkDP",
            "jina_92bcf488db584e2ca21da1e1358fc35emcXW-skRYT-Pp5SwQoRm-Ygv7VUL",
            "jina_80175f92f02644fcb8b6cc97a16d5272L20oG71J0N21mIXXPIywhyMMGDGZ",
            "jina_db0cd9b161994c1bb35e1e58b3af3b6f1TYNCJx7z4fuVM77b3qtpuDVsnRB",
            "jina_7aa5938a901247dfaca8ff7a49c9275d-ZC27fRrqxXqwdZ5xOQVsrPhCVAG",
            "jina_31c0cd97836744bea1951c1ab0d6d95fYO-PrxF69NcxlDI3JiZFMrOs7eYh",
            "jina_38e507f9457645d994d368ff43d8a70a7LbUULQXBSeuvqtV4ekCa3kvK_ko",
            "jina_bb6e02015a774288b207335dfe00731a5-yGaYG5Pt0zQPUd-GCWlj-39dNW"
        ]
        self.token_index = 0

        # Retry mechanism
        self.backoff_factor = 2
        self.retry_delays: Dict[str, float] = {}

        # File paths
        self.failed_urls_file = os.path.join(output_dir, "failed_urls.txt")
        self.url_mapping_file = os.path.join(output_dir, "url_mapping.json")
        self.in_progress_file = os.path.join(output_dir, "in_progress_urls.json")

        # Thread-safe structures
        self.url_queue = Queue()
        self.worker_urls: Dict[str, Tuple[str, int]] = {}

        # Locks
        self.token_lock = threading.Lock()
        self.url_mapping_lock = threading.Lock()
        self.processed_urls_lock = threading.Lock()
        self.id_lock = threading.Lock()
        self.pending_tasks_lock = threading.Lock()
        self.worker_urls_lock = threading.Lock()
        self.retry_delays_lock = threading.Lock()
        self.failed_urls_lock = threading.Lock()

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Setup signal handlers and load existing state
        self.setup_signal_handlers()
        self.last_save = datetime.now()
        self.load_url_mapping()

    def setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received shutdown signal {signum}. Saving state...")
        self.active = False
        self.save_url_mapping()
        self.save_in_progress_urls()
        sys.exit(0)

    def get_next_token(self) -> Optional[str]:
        """Get next available token using round-robin."""
        with self.token_lock:
            if not self.bearer_tokens:
                return None
            token = self.bearer_tokens[self.token_index]
            self.token_index = (self.token_index + 1) % len(self.bearer_tokens)
            return token

    def normalize_url(self, url: str) -> str:
        """Remove hash fragments and normalize URL."""
        try:
            parsed = urlparse(url)
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                ''
            ))
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing URL {url}: {e}")
            return url

    def log_failed_url(self, url: str, error: str) -> None:
        """Log failed URL with error message."""
        with self.failed_urls_lock:
            with open(self.failed_urls_file, 'a') as f:
                f.write(f"{url}\t{error}\t{datetime.now().isoformat()}\n")
        logger.error(f"Added {url} to failed URLs list: {error}")

    def save_in_progress_urls(self):
        """Save currently processing URLs."""
        try:
            with self.worker_urls_lock:
                state = {
                    'in_progress': self.worker_urls,
                    'queue': list(self.url_queue.queue),
                    'timestamp': datetime.now().isoformat()
                }

                temp_file = self.in_progress_file + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump(state, f, indent=2)

                os.replace(temp_file, self.in_progress_file)
                logger.info(f"Saved {len(self.worker_urls)} in-progress URLs")
        except Exception as e:
            logger.error(f"Error saving in-progress URLs: {e}")

    def load_interrupted_state(self):
        """Load and resume interrupted processing."""
        if not os.path.exists(self.in_progress_file):
            return

        try:
            with open(self.in_progress_file) as f:
                state = json.load(f)

            # Resume queued URLs
            for url_data in state.get('queue', []):
                if isinstance(url_data, tuple):
                    url, retry_count = url_data
                else:
                    url, retry_count = url_data, 0
                self.add_url_to_queue(url, retry_count)

            # Resume in-progress URLs
            for thread_id, url_data in state.get('in_progress', {}).items():
                if isinstance(url_data, tuple):
                    url, retry_count = url_data
                else:
                    url, retry_count = url_data, 0

                if retry_count < self.max_retries:
                    self.add_url_to_queue(url, retry_count + 1)
                else:
                    self.log_failed_url(url, "Exceeded max retries across runs")

            logger.info(f"Resumed state from {state.get('timestamp')}")
            os.remove(self.in_progress_file)
        except Exception as e:
            logger.error(f"Error loading interrupted state: {e}")

    def fetch_url_content(self, url: str) -> str:
        """Fetch content with token rotation and error handling."""
        errors = []
        tokens_to_try = self.bearer_tokens.copy()

        while tokens_to_try:
            token = self.get_next_token()
            if not token:
                break

            try:
                headers = {
                    "X-With-Links-Summary": "true",
                    "Authorization": f"Bearer {token}"
                }
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=30
                )

                if response.status_code == 429:  # Rate limit
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise requests.exceptions.RequestException(
                        f"Rate limited, retry after {retry_after}s")

                response.raise_for_status()
                if response.text:
                    return response.text

                errors.append(f"Empty response with token {token[:8]}...")
            except requests.exceptions.RequestException as e:
                errors.append(f"Failed with token {token[:8]}...: {str(e)}")
                tokens_to_try.remove(token)
                continue

        # Try without authentication as last resort
        try:
            headers = {"X-With-Links-Summary": "true"}
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            if response.text:
                return response.text
            errors.append("Empty response without authentication")
        except requests.exceptions.RequestException as e:
            errors.append(f"Failed without authentication: {str(e)}")

        error_message = " | ".join(errors)
        self.log_failed_url(url, error_message)
        raise requests.exceptions.RequestException(f"All attempts failed: {error_message}")

    def process_content(self, content: str) -> Tuple[str, Set[str]]:
        """Process content to extract URLs and clean content."""
        urls = set()
        cleaned_content = content

        try:
            if "Links/Buttons:" in content:
                main_content = content.split("Links/Buttons:")[0].rstrip()
                links_section = content.split("Links/Buttons:")[1].split("\n\n")[0]

                matches = re.finditer(f"({self.base_pattern}[^\s)]+)", links_section)
                for match in matches:
                    url = unquote(match.group(1))
                    if not url.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        normalized_url = self.normalize_url(url)
                        urls.add(normalized_url)

                cleaned_content = main_content

        except Exception as e:
            logger.error(f"Error processing content: {e}")

        return cleaned_content, urls

    def save_content(self, content: str, url: str) -> None:
        """Save content to file with unique ID."""
        try:
            normalized_url = self.normalize_url(url)

            with self.url_mapping_lock:
                if normalized_url not in self.url_mapping:
                    with self.id_lock:
                        self.url_mapping[normalized_url] = self.current_id
                        file_id = self.current_id
                        self.current_id += 1
                else:
                    file_id = self.url_mapping[normalized_url]

            file_path = os.path.join(self.output_dir, f"{file_id}.md")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved content for {normalized_url} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving content for {url}: {e}")

    def load_url_mapping(self, reprocess_urls: Set[str] = None):
        """
        Load existing URL mapping from file if it exists.

        Args:
            reprocess_urls (Set[str], optional): Set of URLs that should be reprocessed
        """
        # reprocess_urls = {'https://www.tradingview.com/charting-library-docs/latest/api/',
        #                   'https://www.tradingview.com/charting-library-docs/latest/getting_started',
        #                   'https://www.tradingview.com/charting-library-docs/latest/connecting_data/UDF'}
        if not os.path.exists(self.url_mapping_file):
            return

        try:
            # Load existing mappings
            with open(self.url_mapping_file, 'r') as f:
                data = json.load(f)

            # Update mapping and ID counter
            with self.url_mapping_lock:
                self.url_mapping = data['url_mapping']
                self.current_id = data['last_id'] + 1

            # Convert reprocess URLs if provided
            normalized_reprocess_urls = set()
            if reprocess_urls:
                normalized_reprocess_urls = {self.normalize_url(url) for url in reprocess_urls}
                logger.info(f"Will reprocess {len(normalized_reprocess_urls)} URLs")

            # Add all URLs to processed except those to be reprocessed
            with self.processed_urls_lock:
                self.processed_urls.update(
                    url for url in self.url_mapping.keys()
                    if url not in normalized_reprocess_urls
                )

            logger.info(f"Loaded {len(self.url_mapping)} URL mappings")
            logger.info(f"Marked {len(self.processed_urls)} URLs as processed")

        except Exception as e:
            logger.error(f"Error loading URL mapping: {e}")

    def save_url_mapping(self):
        """Save URL mapping to file."""
        try:
            with self.url_mapping_lock:
                temp_file = self.url_mapping_file + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump({
                        'url_mapping': self.url_mapping,
                        'last_id': self.current_id - 1
                    }, f, indent=2)
                os.replace(temp_file, self.url_mapping_file)
            logger.info("Saved URL mapping")
        except Exception as e:
            logger.error(f"Error saving URL mapping: {e}")

    def periodic_save(self):
        """Perform periodic state save."""
        now = datetime.now()
        if (now - self.last_save).total_seconds() >= self.save_interval:
            logger.info("Performing periodic state save...")
            self.save_url_mapping()
            self.save_in_progress_urls()
            self.last_save = now

    def add_url_to_queue(self, url: str, retry_count: int = 0, delay: float = 0) -> None:
        """Add URL to queue with optional delay."""
        with self.pending_tasks_lock:
            self.pending_tasks += 1

        if delay > 0:
            timer = threading.Timer(delay, lambda: self.url_queue.put((url, retry_count)))
            timer.daemon = True
            timer.start()
        else:
            self.url_queue.put((url, retry_count))

    def worker(self):
        """Worker function for processing URLs."""
        thread_id = threading.current_thread().name

        while self.active:
            try:
                url_data = self.url_queue.get(timeout=1)
                if isinstance(url_data, tuple):
                    url, retry_count = url_data
                else:
                    url, retry_count = url_data, 0

                with self.worker_urls_lock:
                    self.worker_urls[thread_id] = (url, retry_count)

                try:
                    normalized_url = self.normalize_url(url)

                    with self.processed_urls_lock:
                        if normalized_url in self.processed_urls:
                            with self.pending_tasks_lock:
                                self.pending_tasks -= 1
                            self.url_queue.task_done()
                            continue

                    logger.info(f"Processing {normalized_url} (Attempt {retry_count + 1})")
                    time.sleep(2)  # Rate limiting

                    content = self.fetch_url_content(normalized_url)
                    if content:  # Only process if we got content
                        cleaned_content, new_urls = self.process_content(content)

                        # Only mark as processed after successful processing
                        with self.processed_urls_lock:
                            self.processed_urls.add(normalized_url)

                        self.save_content(cleaned_content, normalized_url)

                        logger.info(f"Found {len(new_urls)} new URLs")
                        for new_url in new_urls:
                            with self.processed_urls_lock:
                                if new_url not in self.processed_urls:
                                    self.add_url_to_queue(new_url, 0)

                        # Clear retry delay on success
                        with self.retry_delays_lock:
                            self.retry_delays.pop(normalized_url, None)

                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")

                    if retry_count < self.max_retries:
                        with self.retry_delays_lock:
                            current_delay = self.retry_delays.get(normalized_url, 1)
                            next_delay = current_delay * self.backoff_factor
                            self.retry_delays[normalized_url] = next_delay

                        logger.info(f"Retrying {url} in {current_delay}s")
                        self.add_url_to_queue(url, retry_count + 1, delay=current_delay)
                    else:
                        logger.error(f"Max retries ({self.max_retries}) reached for {url}")
                        self.log_failed_url(url, f"Max retries reached: {str(e)}")
                        with self.retry_delays_lock:
                            self.retry_delays.pop(normalized_url, None)

                finally:
                    # Clean up worker state
                    with self.worker_urls_lock:
                        self.worker_urls.pop(thread_id, None)
                    self.url_queue.task_done()
                    with self.pending_tasks_lock:
                        self.pending_tasks -= 1

            except Empty:
                if self.pending_tasks == 0:
                    break
                continue

    def crawl(self, start_url: str) -> None:
        """Start parallel crawling process."""
        self.active = True

        # Load any interrupted state before starting
        self.load_interrupted_state()

        # Add start URL only if no state was resumed and queue is empty
        if self.url_queue.empty():
            self.add_url_to_queue(start_url, 0)

        workers = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Start worker threads
            for _ in range(self.num_workers):
                workers.append(executor.submit(self.worker))

            # Wait for all tasks to complete with periodic saving
            try:
                while self.pending_tasks > 0:
                    time.sleep(0.5)
                    self.periodic_save()
            except KeyboardInterrupt:
                logger.info("Stopping crawler...")
                self.active = False
                self.save_url_mapping()
                self.save_in_progress_urls()

            # Wait for all workers to complete
            for future in as_completed(workers):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Worker error: {e}")
                    self.save_url_mapping()
                    self.save_in_progress_urls()

    def get_statistics(self) -> Dict[str, Any]:
        """Get crawler statistics."""
        return {
            'total_processed': len(self.processed_urls),
            'total_mapped': len(self.url_mapping),
            'pending_tasks': self.pending_tasks,
            'active_workers': len(self.worker_urls),
            'current_retry_delays': len(self.retry_delays),
            'last_save_time': self.last_save.isoformat()
        }


def main():
    """Main entry point with error handling and cleanup."""
    start_url = "https://docs.crustdata.com/docs/intro"

    # Create crawler instance
    crawler = DocCrawler(
        num_workers=50,  # Conservative number of workers
        save_interval=60,  # Save every minute
        max_retries=3,
        output_dir="crustdata_docs",
        base_pattern=r"https://docs\.crustdata\.com/docs"
    )

    try:
        # Start crawling
        crawler.crawl(start_url)

        # Get and log final statistics
        stats = crawler.get_statistics()
        logger.info("Crawling completed successfully")
        logger.info(f"Final statistics: {json.dumps(stats, indent=2)}")

    except Exception as e:
        logger.error(f"Fatal error during crawling: {e}")
        # Save state on error
        crawler.save_url_mapping()
        crawler.save_in_progress_urls()
        sys.exit(1)

    finally:
        # Final cleanup
        crawler.save_url_mapping()

        # Print summary of any failed URLs
        if os.path.exists(crawler.failed_urls_file):
            with open(crawler.failed_urls_file, 'r') as f:
                failed_count = sum(1 for _ in f)
            logger.info(f"Total failed URLs: {failed_count}")


def clean_markdown_files(docs_dir: str = "docs") -> None:
    """Clean markdown files by removing specified content."""
    try:
        if not os.path.exists(docs_dir):
            logger.error(f"Directory {docs_dir} does not exist")
            return

        md_files = [f for f in os.listdir(docs_dir) if f.endswith('.md')]
        logger.info(f"Found {len(md_files)} markdown files to process")

        for filename in md_files:
            file_path = os.path.join(docs_dir, filename)
            temp_path = os.path.join(docs_dir, f"temp_{filename}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Remove the Skip to main content section
                pattern = r'\[Skip to main content\]\(.*?\).*?On this page'
                content = re.sub(pattern, 'On this page', content, flags=re.DOTALL)

                # Remove the Docusaurus footer
                content = re.sub(r'Copyright © 2024', '', content)

                # Use atomic write
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                os.replace(temp_path, file_path)

                logger.info(f"Successfully processed {filename}")

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                continue

        logger.info("Completed processing all markdown files")

    except Exception as e:
        logger.error(f"Error during markdown cleaning process: {e}")


if __name__ == "__main__":
    # main()
    clean_markdown_files("crustdata_docs")
