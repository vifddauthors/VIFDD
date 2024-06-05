const puppeteer = require('puppeteer');
const fs = require('fs');
const csv = require('csv-parser');
const { BlobServiceClient } = require('@azure/storage-blob');

// Function to read CSV file and return an array of website names
async function readCSV(filePath) {
  return new Promise((resolve, reject) => {
    const websites = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (row) => {
        const websiteName = row[Object.keys(row)[1]];
        if (websiteName) {
          websites.push(websiteName);
        }
      })
      .on('end', () => {
        resolve(websites);
      })
      .on('error', (error) => {
        reject(error);
      });
  });
}

// Function to take a full page screenshot of a website and save to Azure Blob Storage
/**
 * Takes a screenshot of a given URL and uploads it to Azure Blob Storage.
 * @param {string} url - The URL to take a screenshot of.
 * @param {object} blobServiceClient - The Azure Blob Service client.
 * @param {string} containerName - The name of the container in Azure Blob Storage.
 * @returns {Promise<void>} - A promise that resolves when the screenshot is taken and uploaded successfully, or rejects with an error.
 */
async function takeScreenshotAndUpload(url, blobServiceClient, containerName) {
  try {
    const blobName = `${url.replace(/[^a-zA-Z0-9]/g, '_')}.png`;

    // Check if the blob (screenshot) already exists in the Azure Blob Storage container
    const containerClient = blobServiceClient.getContainerClient(containerName);
    const blobClient = containerClient.getBlobClient(blobName);
    const exists = await blobClient.exists();

    if (exists) {
      console.log(`Screenshot already exists for ${url} in Azure Blob Storage. Skipping.`);
      return;
    }

    const browser = await puppeteer.launch({ args: ['--no-sandbox'], headless: "new" });
    const page = await browser.newPage();
    console.log(`Taking screenshot of ${url}`);
    
    // Wrap this part in a try-catch block
    try {
      await page.goto('https://' + url, { waitUntil: 'networkidle2' });
      const screenshotBuffer = await page.screenshot({ fullPage: true });
      
      // Upload the screenshot to Azure Blob Storage
      const blockBlobClient = containerClient.getBlockBlobClient(blobName);
      await blockBlobClient.upload(screenshotBuffer, screenshotBuffer.length);
      
      console.log(`Screenshot uploaded to Azure Blob Storage: ${blobName}`);
    } catch (error) {
      console.error(`Error taking/uploading screenshot of ${url}: ${error.message}`);
    } finally {
      // Close the browser regardless of whether there's an error or not
      await browser.close();
    }
  } catch (error) {
    console.error(`Error in takeScreenshotAndUpload function for ${url}: ${error.message}`);
  }
}

// Main function
async function main() {
  const csvFilePath = 'csv-file.csv'; // Change to the actual path of your CSV file
  const websites = await readCSV(csvFilePath);

  // Azure Blob Storage configuration
  const blobServiceClient = BlobServiceClient.fromConnectionString('your-connection-string-here');
  const containerName = 'container-name-here';

  for (const website of websites) {
    await takeScreenshotAndUpload(website, blobServiceClient, containerName);
  }
}

main();
