import fetch from 'node-fetch';
import { writeFileSync } from 'fs';

const apiUrl = 'https://inspirehep.net/api/literature';
const queryParams = new URLSearchParams({
    sort: 'mostrecent',
    size: '25',
    page: '1',
    q: 'find author e.witten.1 or dillon.j.brout.1',
    fields: 'titles,authors.full_name,created'
}).toString();

async function fetchPapers() {
    const response = await fetch(`${apiUrl}?${queryParams}`);
    const data = await response.json();
    return data.hits.hits;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toUTCString();
}

function createRSSFeed(papers) {
    const rssItems = papers.map(paper => {
        const title = paper.metadata.titles[0].title;
        const authors = paper.metadata.authors.map(author => author.full_name).slice(0, 3).join(', ') + (paper.metadata.authors.length > 3 ? ', et al.' : '');
        const publicationDate = formatDate(paper.created);
        const paperUrl = `https://inspirehep.net/literature/${paper.id}`;
        return `
            <item>
                <title><![CDATA[${title}]]></title>
                <link>${paperUrl}</link>
                <dc:creator><![CDATA[${authors}]]></dc:creator>
                <pubDate>${publicationDate}</pubDate>
            </item>
        `;
    }).join('');

    return `<?xml version="1.0" encoding="UTF-8" ?>
        <rss version="2.0">
            <channel>
                <title>Papers</title>
                ${rssItems}
            </channel>
        </rss>
    `;
}

async function main() {
    const papers = await fetchPapers();
    const rssFeed = createRSSFeed(papers);
    writeFileSync('feed.xml', rssFeed, 'utf8');
}

main().catch(console.error);
