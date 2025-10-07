#!/usr/bin/env python3
"""Command-line interface for AI Outlier Detection Pipeline."""

import click
import os
import json
from pathlib import Path
from src.pipeline import AIOutlierDetectionPipeline

@click.group()
def cli():
    """AI-Powered Outlier Detection Pipeline CLI"""
    pass

@cli.command()
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--output', '-o', default='results', help='Output directory')
@click.option('--no-save', is_flag=True, help='Skip saving results to disk')
def run(config, output, no_save):
    """Run the complete outlier detection pipeline."""
    click.echo("🚀 Starting AI Outlier Detection Pipeline...")
    
    try:
        pipeline = AIOutlierDetectionPipeline(config)
        results = pipeline.run_full_pipeline(
            save_results=not no_save,
            output_dir=output
        )
        
        click.echo("\n📊 Pipeline Results:")
        click.echo(f"Total documents processed: {results['total_documents']}")
        
        for method, count in results['summary_stats'].items():
            click.echo(f"{method}: {count} outliers detected")
        
        if not no_save:
            click.echo(f"\n💾 Results saved to: {output}/")
        
        click.echo("✅ Pipeline completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('texts', nargs=-1, required=True)
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--category', help='Category label for texts')
def detect(texts, config, category):
    """Detect outliers in custom text inputs."""
    click.echo(f"🔍 Analyzing {len(texts)} text(s)...")
    
    try:
        pipeline = AIOutlierDetectionPipeline(config)
        categories = [category] * len(texts) if category else None
        
        results = pipeline.detect_outliers_in_text(list(texts), categories)
        
        click.echo("\n📊 Outlier Detection Results:")
        for method, data in results.items():
            click.echo(f"\n{method}:")
            click.echo(f"  Outliers found: {data['outlier_count']}")
            if data['outlier_indices']:
                click.echo(f"  Outlier indices: {data['outlier_indices']}")
        
    except Exception as e:
        click.echo(f"❌ Detection failed: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--text-column', default='text', help='Column name containing text data')
@click.option('--category-column', help='Column name containing categories')
def analyze_file(file_path, config, text_column, category_column):
    """Analyze outliers in a CSV/JSON file."""
    import pandas as pd
    
    click.echo(f"📄 Loading data from: {file_path}")
    
    try:
        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in file")
        
        texts = df[text_column].tolist()
        categories = df[category_column].tolist() if category_column and category_column in df.columns else None
        
        pipeline = AIOutlierDetectionPipeline(config)
        results = pipeline.detect_outliers_in_text(texts, categories)
        
        click.echo(f"\n📊 Analysis Results for {len(texts)} documents:")
        for method, data in results.items():
            click.echo(f"\n{method}:")
            click.echo(f"  Outliers found: {data['outlier_count']}")
            if data['outlier_indices']:
                click.echo(f"  Outlier indices: {data['outlier_indices'][:10]}...")  # Show first 10
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        raise click.Abort()

@cli.command()
def setup():
    """Set up the environment and configuration files."""
    click.echo("🔧 Setting up AI Outlier Detection Pipeline...")
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        click.echo("Creating .env file...")
        with open('.env', 'w') as f:
            f.write("# Copy from .env.example and fill in your API key\n")
            f.write("NEBIUS_API_KEY=your_api_key_here\n")
        click.echo("📝 Created .env file. Please add your API key.")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    click.echo("📁 Created results directory")
    
    click.echo("✅ Setup completed!")
    click.echo("\n📋 Next steps:")
    click.echo("1. Add your API key to .env file")
    click.echo("2. Run: python cli.py run")

if __name__ == '__main__':
    cli()