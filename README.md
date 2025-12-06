# Semantic Search

A hybrid semantic search engine for book discovery that combines traditional keyword matching with modern semantic understanding to deliver highly relevant search results.

## Overview

This project implements a hybrid semantic search system designed to search through book collections using both traditional and semantic approaches. The system understands the context and meaning behind search queries, going beyond simple keyword matching to find books that are semantically similar to user queries.

## Features

- **Hybrid Search Approach**: Combines traditional keyword-based search with semantic similarity
- **Book Database**: Searches through a comprehensive collection of books with metadata
- **Semantic Understanding**: Uses advanced NLP techniques to understand query intent
- **Fast Retrieval**: Optimized for quick search results even with large datasets
- **Relevance Ranking**: Intelligently ranks results based on semantic similarity and relevance

## Technology Stack

- **Python**: Core programming language
- **Natural Language Processing**: For semantic understanding and text embeddings
- **Vector Search**: For efficient similarity computation
- **CSV Data Storage**: Lightweight data management with Book_Details.csv

## Project Structure

```
Semantic-Search/
├── SemanticSearch.py       # Main semantic search implementation
├── Book_Details.csv        # Book database with metadata
└── README.md              # Project documentation
```

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn sentence-transformers
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Bharathsahil-07/Semantic-Search.git
cd Semantic-Search
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run the semantic search engine:

```python
python SemanticSearch.py
```

### Example Queries

The system can handle various types of queries:
- Simple keyword searches: "python programming"
- Semantic queries: "books about artificial intelligence and machine learning"
- Context-aware searches: "mystery novels with unexpected plot twists"

## How It Works

1. **Data Loading**: Reads book information from Book_Details.csv
2. **Text Processing**: Preprocesses and cleans book descriptions and metadata
3. **Embedding Generation**: Creates vector representations of book content
4. **Query Processing**: Converts user queries into embeddings
5. **Similarity Matching**: Computes semantic similarity between query and books
6. **Result Ranking**: Returns top matches ranked by relevance

## Data Format

The `Book_Details.csv` file contains book information with fields such as:
- Title
- Author
- Description
- Genre/Category
- Additional metadata

## Key Algorithms

- **Semantic Embeddings**: Transforms text into high-dimensional vectors
- **Cosine Similarity**: Measures semantic similarity between queries and documents
- **Hybrid Scoring**: Combines traditional TF-IDF with semantic similarity

## Performance

The hybrid approach offers:
- Better handling of synonyms and related concepts
- Improved results for natural language queries
- Balanced precision and recall compared to keyword-only search

## Future Enhancements

- [ ] Add support for multi-language search
- [ ] Implement query expansion and suggestion
- [ ] Add filtering by author, genre, and publication date
- [ ] Create web interface for easier interaction
- [ ] Integrate with external book APIs
- [ ] Implement user feedback mechanism for relevance tuning

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

Bharath Sahil - [@Bharathsahil-07](https://github.com/Bharathsahil-07)

Project Link: [https://github.com/Bharathsahil-07/Semantic-Search](https://github.com/Bharathsahil-07/Semantic-Search)

## Acknowledgments

- Inspired by modern information retrieval techniques
- Built with open-source NLP libraries
- Thanks to the semantic search research community

---

**Note**: This is a demonstration project showcasing hybrid semantic search capabilities for educational and research purposes.
