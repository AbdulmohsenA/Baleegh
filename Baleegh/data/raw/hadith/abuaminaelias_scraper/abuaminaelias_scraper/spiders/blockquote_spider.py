import scrapy
from urllib.parse import urljoin

class BlockquoteSpider(scrapy.Spider):
    name = 'blockquote_spider'
    allowed_domains = ['abuaminaelias.com']
    start_urls = ['https://www.abuaminaelias.com/']

    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'DEFAULT_REQUEST_HEADERS': {
            'Referer': 'https://www.abuaminaelias.com/',
        },
    }

    def parse(self, response):
        # Collect all blockquote tags and their inner HTML
        blockquotes = response.css('blockquote')
        for blockquote in blockquotes:
            # Get the inner HTML of the blockquote
            inner_html = blockquote.get()
            yield {'blockquote': inner_html.strip()}

        # Follow links to other pages only if they are internal
        for href in response.css('a::attr(href)').getall():
            # Create an absolute URL
            absolute_url = urljoin(response.url, href)
            # Check if the URL is internal
            if self.is_internal_link(absolute_url):
                yield response.follow(absolute_url, self.parse)

    def is_internal_link(self, url):
        """Check if a URL is internal (within the allowed domain)."""
        return url.startswith('https://www.abuaminaelias.com/') or url.startswith('http://www.abuaminaelias.com/')
