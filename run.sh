#!/bin/bash
# ============================================================
# RAG Semantic Cache - Quick Start Script
# ============================================================

set -e  # Exit on error

echo "🚀 RAG Semantic Cache API - Quick Start"
echo "========================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating from .env.example..."
    cp .env.example .env
    echo ""
    echo "⚠️  Please edit .env and add your GROQ_API_KEY"
    echo "Get your free key at: https://console.groq.com/keys"
    echo ""
    read -p "Press Enter after you've updated .env..."
fi

# Ask user for deployment method
echo "Choose deployment method:"
echo "1) Docker Compose (recommended)"
echo "2) Local Python"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    echo ""
    echo "🐳 Starting with Docker Compose..."
    echo ""
    
    # Build and start
    docker compose up -d --build
    
    echo ""
    echo "✅ Services started!"
    echo ""
    echo "📚 API Documentation: http://localhost:8000/docs"
    echo "🔍 Health Check: http://localhost:8000/api/v1/health"
    echo ""
    echo "View logs: docker compose logs -f"
    echo "Stop services: docker compose down"
    
elif [ "$choice" == "2" ]; then
    echo ""
    echo "🐍 Starting with Local Python..."
    echo ""
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo "Python version: $python_version"
    
    # Create virtual environment if doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    
    # Check if Redis is running
    if ! redis-cli ping &> /dev/null; then
        echo ""
        echo "⚠️  Redis not running!"
        echo "Start Redis in another terminal:"
        echo "  redis-server"
        echo ""
        read -p "Press Enter when Redis is running..."
    fi
    
    # Start application
    echo ""
    echo "Starting application..."
    python app/main.py
    
else
    echo "Invalid choice!"
    exit 1
fi