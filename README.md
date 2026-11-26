

# 🛡️ ARP Spoofing Detection Dashboard

[![View in Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://arp-spoofing-detection.streamlit.app/)

**Live Demo:** [https://arp-spoofing-detection.streamlit.app/](https://arp-spoofing-detection.streamlit.app/)

## What is ARP? What is ARP Spoofing?

**Address Resolution Protocol (ARP)** is a fundamental protocol in computer networks that maps IP addresses to physical MAC addresses, allowing devices to communicate within a local network. However, ARP is inherently insecure and susceptible to attacks.

**ARP Spoofing** (or ARP Poisoning) is a cyberattack where a malicious actor sends fake ARP messages onto a network. This tricks devices into associating the attacker’s MAC address with the IP address of another device (such as the gateway), enabling the attacker to intercept, modify, or block network traffic. ARP spoofing is a common technique used in Man-in-the-Middle (MitM) attacks and can lead to data theft, session hijacking, or denial of service.

---

## Project Overview

This project provides a comprehensive, interactive dashboard for detecting ARP spoofing attacks using machine learning. Built with Streamlit, it enables:

- **Data exploration and visualization**
- **Training and evaluation of multiple ML models**
- **Real-time predictions on custom network data**
- **Side-by-side model comparison**

The dashboard is designed for network engineers, security researchers, students, and anyone interested in network security and ML.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
python -m streamlit run app/app.py
```


### 3. Access the Dashboard
Open your browser and go to: `http://localhost:8501`

---

### 🌐 Try the Live Demo

[![View in Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://arp-spoofing-detection.streamlit.app/)

**Live Demo:** [https://arp-spoofing-detection.streamlit.app/](https://arp-spoofing-detection.streamlit.app/)

---

## 📊 Dashboard Features

1. **Data Overview** 📊
   - Dataset statistics and sample counts
   - Label distribution visualization
   - Feature importance analysis
   - Data quality metrics
2. **Model Training** 🤖
   - Train multiple ML models: Random Forest, XGBoost, SVM, Isolation Forest, Autoencoder
   - Real-time performance metrics
   - Model comparison table
3. **Visualizations** 📈
   - Interactive confusion matrices
   - ROC curves with AUC scores
   - t-SNE data visualization
   - Feature distribution plots
4. **Real-time Predictions** 🔮
   - Input custom network traffic features
   - Get predictions from all trained models
   - Ensemble voting system
   - Confidence scores
5. **Model Comparison** 📋
   - Side-by-side performance metrics
   - Interactive comparison charts
   - Model recommendations
   - Best model identification

---

## 🔧 Technical Details

### Features Used
- **Rate**: Packet rate per second
- **ARP**: ARP protocol flag (0/1)
- **IAT**: Inter-Arrival Time between packets

### Models Implemented
1. **Random Forest**: Ensemble of decision trees
2. **XGBoost**: Gradient boosting framework
3. **SVM**: Support Vector Machine with RBF kernel
4. **Isolation Forest**: Unsupervised anomaly detection
5. **Autoencoder**: Neural network for reconstruction error

### Data Preprocessing
- Heuristic-based labeling using quantile thresholds
- Standard scaling for feature normalization
- Handling of infinite and missing values
- PCA for autoencoder dimensionality reduction

---

## 📁 File Structure


```
arp-spoofing-detection/
├── app/app.py                 # Streamlit entry point
├── core/                      # Core ML and network utilities
├── data/                      # Training and test data
├── requirements.txt           # Python dependencies
├── tests/                     # Test modules


└── README.md                  # Project documentation
```

---

## 🎯 Usage Examples

### Training Models
1. Navigate to "🤖 Model Training" page
2. Click "Train All Models" button
3. View performance metrics table

### Making Predictions
1. Go to "🔮 Predictions" page
2. Input your network traffic features:
   - Rate: Packet rate (e.g., 100.0)
   - ARP: ARP protocol flag (0 or 1)
   - IAT: Inter-arrival time (e.g., 0.01)
3. View predictions from all models

### Analyzing Results
1. Check "📈 Visualizations" for confusion matrices and ROC curves
2. Use "📋 Model Comparison" to compare model performance
3. Review "📊 Data Overview" for dataset insights

---

## 🔍 Model Performance Metrics

The dashboard automatically calculates and displays:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

---

## 🛠️ Customization

### Adding New Models
1. Add model training code in `train_models()` function
2. Include model in the models dictionary
3. Add prediction logic in the predictions section

### Modifying Features
1. Update the `features` list in `preprocess_data()`
2. Adjust the labeling heuristics in `label_row()`
3. Update the prediction input interface

### Styling
- Modify the CSS in the `st.markdown()` section
- Update colors and layout in Plotly figures
- Customize the page configuration

---

## 🐛 Troubleshooting

### Common Issues
1. **Data not loading**: Ensure CSV files are in the correct directory
2. **Models not training**: Check that all dependencies are installed
3. **Visualizations not showing**: Verify Plotly is installed correctly

### Performance Tips
- Use the cached functions for better performance
- Reduce t-SNE computation for large datasets
- Consider model selection for faster predictions

---

## 📈 Future Enhancements

- Real-time data streaming integration
- Model persistence and loading
- Advanced feature engineering
- Hyperparameter tuning interface
- Export functionality for results
- Alert system for detected attacks

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- Dataset: ARP spoofing detection dataset
- Libraries: Streamlit, scikit-learn, XGBoost, Plotly
- Inspiration: Network security and machine learning community

---

## 📞 Contact

If you have any questions or suggestions, please feel free to open an issue or contact the maintainers.

---

⭐ **Star this repository if you found it helpful!**

## Evaluation limitations

Heuristic labels are not independent attack ground truth. Reported metrics measure agreement with the labeling procedure and do not establish real-world detection accuracy. Use only authorized network data.
