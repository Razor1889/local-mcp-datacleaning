# A Simple MCP Data Cleaner

An MCP server that pairs with Claude Desktop to clean CSV files.
Additionally offers the ability to manipulate cleaning parameters by
editing server.py.

Since it uses Claude Desktop, it CAN be used for FREE using Claude's Free tier.

Usage:
  Upload a messy CSV to Claude
  Prompt Claude to use the available cleaning tools to clean the CSV.
  Claude reponds with a simple cleaning analysis, asking for user input where needed.

## Setup

### Install Dependencies

```bash
pip install pandas numpy mcp
```

### Configure Claude Desktop

Edit the Claude Desktop Config file at:
  Windows: `%APPDATA%\Claude\claude_desktop_config.json`
  Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`

Or Alternatively:
  Navigate to Claude Desktop Settings -> Developer -> Edit Config


Add this to the config file:
```json
{
    "mcpServers": {
      "data-cleaner": {
        "command": "python",
        "args": ["path/DataCleaner/server.py"]
      }
    }
  }
```
Replace `path` with your full path.

### Restart Claude Desktop

Close and reopen Claude Desktop app. The MCP server should run automatically, click on Connections and data-cleaner should be toggled on.

### Testing

Prompt Claude with:

Do you have access to data cleaning tools?

Claude responds with something like:

Yes! I have three data cleaning tools:
- analyze_data_quality
- suggest_cleaning  
- execute_cleaning


## Usage Examples

### Example: Basic Cleaning

You: I have a messy sales CSV. Can you clean it for me?

Claude: I'll help you clean your data. First, let me analyze 
the quality issues.

[Claude calls analyze_data_quality]

I found:
- 15 missing values in the 'price' column
- 3 duplicate rows
- 2 outliers in 'quantity'

Let me get cleaning recommendations...

[Claude calls suggest_cleaning]

I recommend:
1. Fill missing prices with median value
2. Remove duplicate rows
3. Review the outliers before removing

Shall I proceed with cleaning?

You: Yes, please clean it

Claude: [calls execute_cleaning]

Done! I've created:
- cleaned_sales.csv (clean data)
- cleaned_sales_report.txt (detailed report)

Summary:
- Original: 100 rows
- Cleaned: 97 rows (3 duplicates removed)
- All missing values filled
- Data is now ready for analysis!

