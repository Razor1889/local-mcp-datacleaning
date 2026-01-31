'''
A simple MCP Data Cleaner Server

An MCP server that gives Claude the ability to clean CSV files.

Currently incorporated with Claude Desktop, but can be adapted for other MCP clients.

Usage with Claude Desktop:
1. Add this to your MCP config in Claude Desktop (claude_desktop_json config):
    %APPDATA%\Claude\claude_desktop_config.json
2. Ensure that the server workspace contains dependencies outlined in requirements.txt

Read the README.md for detailed instructions.

Tools provided:
- analyze_data_quality
- suggest_cleaning
- execute_cleaning -> based on suggestions

Ideal use case: Data scientists or analysts wanting to automate data cleaning tasks using Claude.
Ideal tool order: analyze_data_quality -> suggest_cleaning -> execute_cleaning
'''

import asyncio
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("data-cleaner")

# Tool Definitions:
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="analyze_data_quality",
            description="""
            Analyzes a CSV file to identify data quality issues:
            - Missing values (count per col)
            - Duplicate rows
            - Potential outliers in numeric columns
            - Data type issues (numbers stored as text etc)
            - Quality score (0-100)
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    }
                },
                "required": ["file_path"]
            }
        ),
        
        Tool(
            name="suggest_cleaning",
            description="""
            Suggests specific cleaning operations based on the data quality issues.
            
            Returns recommendations like:
            - "Fill missing values in 'col' with median"
            - "Remove 3 duplicate rows"
            - "Convert 'quantity' from text to numeric"
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    }
                },
                "required": ["file_path"]
            }
        ),
        
        Tool(
            name="execute_cleaning",
            description="""
            Cleans the CSV file based on recommendations.
            
            Saves a cleaned version and generates a report showing:
            - What was changed
            - How many rows/columns were affected
            - Before and after stats
            
            Returns the path to the cleaned file and the report.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the messy CSV file"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Where to save the cleaned CSV (optional)"
                    },
                    "remove_outliers": {
                        "type": "boolean",
                        "description": "Whether to remove outliers (default: false for manual review)"
                    }
                },
                "required": ["file_path"]
            }
        )
    ]


# Tool Implementations:
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from Claude."""
    try:
        if name == "analyze_data_quality":
            result = await analyze_data_quality(arguments["file_path"])
            
        elif name == "suggest_cleaning":
            result = await suggest_cleaning(arguments["file_path"])
            
        elif name == "execute_cleaning":
            result = await execute_cleaning(
                arguments["file_path"],
                arguments.get("output_path"),
                arguments.get("remove_outliers", False)
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
        
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "error_type": type(e).__name__
            }, indent=2)
        )]

# Analyze Data Quality Tool:
async def analyze_data_quality(file_path: str) -> dict:
    df = pd.read_csv(file_path)
    
    # missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].to_dict()
    
    # dups
    duplicates = df.duplicated().sum()
    
    # outliers in numeric columns
    outliers_info = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outlier_count > 0:
            outliers_info[col] = int(outlier_count)
    
    # data type issues (numbers stored as text)
    type_issues = []
    for col in df.select_dtypes(include=['object']).columns:
        try:
            pd.to_numeric(df[col].dropna(), errors='raise')
            type_issues.append(col)
        except:
            pass
    
    # quality score
    # completeness (40), uniqueness (30), validity (30)
    # comepletness: (1 - % missing) * 40
    # uniqueness: (1 - % duplicates) * 30
    # validity: 30 - (number of type issues * 5)
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 40
    uniqueness = (1 - duplicates / len(df)) * 30
    validity = 30 - (len(type_issues) * 5)
    quality_score = max(0, completeness + uniqueness + validity)
    
    return {
        "file": file_path,
        "rows": len(df),
        "columns": len(df.columns),
        "quality_score": round(quality_score, 1),
        "issues": {
            "missing_values": {
                "total_cells": int(df.isnull().sum().sum()),
                "affected_columns": {k: int(v) for k, v in missing_cols.items()}
            },
            "duplicates": int(duplicates),
            "outliers": outliers_info,
            "type_issues": type_issues
        },
        "summary": f"Quality Score: {quality_score:.1f}/100. Found {len(missing_cols)} columns with missing values, {duplicates} duplicate rows, and outliers in {len(outliers_info)} columns."
    }

# Suggest Cleaning Tool:
async def suggest_cleaning(file_path: str) -> dict:
    df = pd.read_csv(file_path)
    suggestions = []
    
    # Missing values
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        missing_pct = (missing[col] / len(df)) * 100
        
        if missing_pct > 50:
            suggestions.append({
                "action": "drop_column",
                "column": col,
                "reason": f"{missing_pct:.1f}% missing - too much to impute reliably"
            })
        elif pd.api.types.is_numeric_dtype(df[col]):
            suggestions.append({
                "action": "fill_with_median",
                "column": col,
                "reason": f"Numeric column with {int(missing[col])} missing values"
            })
        else:
            suggestions.append({
                "action": "fill_with_mode",
                "column": col,
                "reason": f"Categorical column with {int(missing[col])} missing values"
            })
    
    # Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        suggestions.append({
            "action": "remove_duplicates",
            "count": int(duplicates),
            "reason": "Exact duplicate rows found"
        })
    
    # Type conversions
    for col in df.select_dtypes(include=['object']).columns:
        try:
            pd.to_numeric(df[col].dropna(), errors='raise')
            suggestions.append({
                "action": "convert_to_numeric",
                "column": col,
                "reason": "Values are numeric but stored as text"
            })
        except:
            pass
    
    # Outliers
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outlier_count > 0:
            suggestions.append({
                "action": "review_outliers",
                "column": col,
                "count": int(outlier_count),
                "reason": "Statistical outliers detected - review before removing"
            })
    
    return {
        "file": file_path,
        "total_suggestions": len(suggestions),
        "suggestions": suggestions
    }

# Execute Cleaning Tool:
async def execute_cleaning(file_path: str, output_path: str = None, remove_outliers: bool = False) -> dict:
    df = pd.read_csv(file_path)
    original_shape = df.shape
    
    report = []
    report.append("=" * 60)
    report.append("DATA CLEANING REPORT")
    report.append(f"File: {file_path}")
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append(f"\nOriginal: {original_shape[0]} rows × {original_shape[1]} columns")
    
    # 1. Remove duplicates
    before_dup = len(df)
    df = df.drop_duplicates()
    after_dup = len(df)
    if before_dup - after_dup > 0:
        report.append(f"\n✓ Removed {before_dup - after_dup} duplicate rows")
    
    # 2. Handle missing values
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        missing_pct = (missing[col] / len(df)) * 100
        
        if missing_pct > 50:
            df = df.drop(columns=[col])
            report.append(f"✓ Dropped column '{col}' ({missing_pct:.1f}% missing)")
        elif pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            report.append(f"✓ Filled '{col}' with median ({median_val:.2f})")
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
            df[col].fillna(mode_val, inplace=True)
            report.append(f"✓ Filled '{col}' with mode ('{mode_val}')")
    
    # Fix data types
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
            report.append(f"✓ Converted '{col}' to numeric")
        except:
            pass
    
    # Remove outliers (if requested)
    if remove_outliers:
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            before_outlier = len(df)
            df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
            after_outlier = len(df)
            if before_outlier - after_outlier > 0:
                report.append(f"✓ Removed {before_outlier - after_outlier} outliers from '{col}'")
    
    # Save cleaned data
    if output_path is None:
        output_path = file_path.replace('.csv', '_cleaned.csv')
    
    df.to_csv(output_path, index=False)
    
    final_shape = df.shape
    report.append(f"\n{'=' * 60}")
    report.append(f"Final: {final_shape[0]} rows × {final_shape[1]} columns")
    report.append(f"Rows removed: {original_shape[0] - final_shape[0]}")
    report.append(f"Columns removed: {original_shape[1] - final_shape[1]}")
    report.append(f"\n✓ Saved to: {output_path}")
    
    # Save report
    report_path = output_path.replace('.csv', '_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    return {
        "success": True,
        "original_shape": original_shape,
        "cleaned_shape": final_shape,
        "cleaned_file": output_path,
        "report_file": report_path,
        "report": '\n'.join(report)
    }



async def main():
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
