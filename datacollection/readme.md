# README

The `datacollection` folder in the VIFDD-2024 repository contains scripts and resources related to the data collection process. This includes capturing full-page screenshots of websites and uploading them to Azure Blob Storage. Below is a detailed guide on how to use the `screenshot.js` script and set up the necessary components.

## Screenshot.js

The `screenshot.js` script utilizes Puppeteer, a Node library for headless browser automation, to capture full-page screenshots of websites. The screenshots are then uploaded to Azure Blob Storage for further analysis. 

### Prerequisites

1. **Node.js and npm**: Ensure that Node.js and npm (Node Package Manager) are installed on your machine. You can download them [here](https://nodejs.org/).

2. **Azure Storage Account**: Create an Azure Storage Account and note down the connection string. You can create an account [here](https://portal.azure.com/).

### Installation

1. Open a terminal and navigate to the `datacollection` folder:
   ```bash
   cd path/to/VIFDD/datacollection
   ```

2. Install the required Node.js packages:
   ```bash
   npm install puppeteer fs csv-parser @azure/storage-blob
   ```

### Configuration

1. **CSV File**: Replace `'csv-file.csv'` in the script with the actual path to your CSV file containing website names.

2. **Azure Blob Storage**: Replace `'your-connection-string-here'` with your Azure Storage Account connection string and `'container-name-here'` with the name of your Azure Blob Storage container.

### Execution

Run the script using the following command:

```bash
node screenshot.js
```

The script will iterate through the websites listed in the CSV file, capture full-page screenshots, and upload them to Azure Blob Storage.

*Note: Ensure that Puppeteer's `launch` method has the appropriate options for your environment, as it may vary based on your system configuration.*

### Data Collection Workflow

