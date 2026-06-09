import { NextResponse } from 'next/server';
import puppeteer from 'puppeteer';
import { logger } from '@/lib/utils';

// Docker-safe Chrome launch args
const CHROME_ARGS = [
  '--no-sandbox',
  '--disable-setuid-sandbox',
  '--disable-dev-shm-usage',   // Critical: prevents /dev/shm overflow in containers
  '--disable-gpu',
  '--no-first-run',
  '--no-zygote',
  '--disable-extensions',
  '--disable-background-networking',
  '--disable-default-apps',
  '--mute-audio',
];

// Patterns that indicate an audio stream URL
function isAudioUrl(url: string): boolean {
  const lower = url.toLowerCase();
  return (
    lower.includes('.mp3') ||
    lower.includes('.m4a') ||
    lower.includes('.aac') ||
    lower.includes('.ogg') ||
    lower.includes('.wav') ||
    lower.includes('.flac') ||
    lower.includes('/audio/') ||
    lower.includes('audio-') ||
    lower.includes('podcast') ||
    // Xiaoyuzhou CDN patterns
    lower.includes('xyzcdn.net') ||
    lower.includes('xiaoyuzhoufm') && lower.includes('media') ||
    lower.includes('typlog') ||
    lower.includes('chtbl.com') ||  // common podcast tracking domain
    lower.includes('cdn') && (lower.includes('.mp3') || lower.includes('.m4a'))
  );
}

export async function POST(request: Request) {
  let browser;
  try {
    const { url } = await request.json();

    if (!url || !url.includes('xiaoyuzhoufm.com')) {
      return NextResponse.json(
        { error: 'Invalid Xiaoyuzhou URL' },
        { status: 400 }
      );
    }

    logger.info('[ParseURL] Launching browser for:', url);

    browser = await puppeteer.launch({
      headless: true,
      executablePath: process.env.PUPPETEER_EXECUTABLE_PATH,
      args: CHROME_ARGS,
    });

    const page = await browser.newPage();

    // Block unnecessary resources to speed up loading
    await page.setRequestInterception(true);

    const audioUrl = await new Promise<string>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error('Timeout: Audio URL not found within 45s'));
      }, 45000);

      page.on('request', (req) => {
        const reqUrl = req.url();
        const resourceType = req.resourceType();

        // Check URL patterns first (catches audio before content-type is known)
        if (isAudioUrl(reqUrl)) {
          logger.info('[ParseURL] Found audio URL via pattern:', reqUrl);
          clearTimeout(timeoutId);
          resolve(reqUrl);
          req.continue();
          return;
        }

        // Also catch by resource type
        if (resourceType === 'media') {
          logger.info('[ParseURL] Found audio URL via media type:', reqUrl);
          clearTimeout(timeoutId);
          resolve(reqUrl);
          req.continue();
          return;
        }

        // Block heavy resources we don't need
        if (['image', 'font', 'stylesheet'].includes(resourceType)) {
          req.abort();
          return;
        }

        req.continue();
      });

      page.goto(url, {
        waitUntil: 'networkidle2',  // less strict than networkidle0
        timeout: 40000,
      }).catch(reject);
    });

    logger.info('[ParseURL] Successfully extracted audio URL');
    return NextResponse.json({ audioUrl });

  } catch (error) {
    logger.error('[ParseURL] Error:', error);
    return NextResponse.json(
      { error: 'Failed to parse Xiaoyuzhou URL' },
      { status: 500 }
    );
  } finally {
    if (browser) {
      await browser.close().catch(() => {});
    }
  }
}
