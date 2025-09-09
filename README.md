# 🛡️ ARP Spoofing Detection Dashboard

A comprehensive machine learning-based system for detecting ARP spoofing attacks in network traffic data, featuring an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Features

- **Interactive Dashboard**: Beautiful Streamlit web interface
- **Multiple ML Models**: Random Forest, XGBoost, SVM, Isolation Forest, Autoencoder
- **Real-time Predictions**: Input custom network features and get instant results
- **Advanced Visualizations**: Interactive confusion matrices, ROC curves, t-SNE plots
- **Model Comparison**: Side-by-side performance analysis
- **Data Analysis**: Comprehensive dataset overview and statistics

## 📊 Dashboard Preview

The dashboard includes 5 main sections:
- **📊 Data Overview**: Dataset statistics and feature analysis
- **🤖 Model Training**: Train and evaluate multiple ML models
- **📈 Visualizations**: Interactive charts and plots
- **🔮 Predictions**: Real-time ARP spoofing detection
- **📋 Model Comparison**: Performance metrics comparison

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ARWA044/arp-spoofing-detection.git
   cd arp-spoofing-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   python run_dashboard.py
   ```
   Or directly with Streamlit:
   ```bash
   streamlit run streamlit_dashboard.py
   ```

4. **Access the dashboard:**
   Open your browser and go to: `http://localhost:8501`

## 📁 Project Structure

```
arp-spoofing-detection/
├── streamlit_dashboard.py      # Main dashboard application
├── run_dashboard.py           # Dashboard launcher script
├── requirements.txt           # Python dependencies
├── ARP_Spoofing_train.pcap.csv # Training data
├── ARP_Spoofing_test.pcap.csv  # Test data
├── README.md                  # This file
└── DASHBOARD_README.md        # Detailed dashboard documentation
```

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

## 📈 Usage Examples

### Quick Start
1. Run the dashboard: `python run_dashboard.py`
2. Navigate to "🤖 Model Training" and click "Train All Models"
3. Go to "🔮 Predictions" to test with custom data
4. Check "📈 Visualizations" for model performance analysis

### Making Predictions
Input your network traffic features:
- **Rate**: Packet rate (e.g., 100.0)
- **ARP**: ARP protocol flag (0 or 1)
- **IAT**: Inter-arrival time (e.g., 0.01)

The dashboard will show predictions from all trained models with confidence scores.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: ARP spoofing detection dataset
- Libraries: Streamlit, scikit-learn, XGBoost, Plotly
- Inspiration: Network security and machine learning community

## 📞 Contact

If you have any questions or suggestions, please feel free to open an issue or contact the maintainers.

---

⭐ **Star this repository if you found it helpful!**
