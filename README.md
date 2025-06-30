# Drone-based Herbicide Dye Detection - Colab Experiments

## Overview

Drone imagery is increasingly used to map weed plants with computer vision systems. Ensuring alignment between treatments and weed maps is critical as untreated weeds escape management. Conversely, over-spray of herbicides increases chemical costs and environmental impacts. Yet, practitioners lack automated methods to audit the extent of herbicide treatments. We developed a system to detect herbicide dyes using drones equipped only with visible light sensors. We tested the system across seasons (May, July, October), application rates, patch sizes, and colors in a grassland ecosystem. To augment data diversity and amount we overlaid colored patches on unsprayed images during model training and evaluated the benefit of including these synthetic data. Overall detection performance was excellent for both dye colors (F1-score 0.979 for red trained on mixed real and synthetic data; 0.929 for blue trained on real only), but blue dyes were more difficult to detect during summer. Models trained only on synthetic data performed well for real-world treatments, with red dye (F1 0.962) outperforming blue (F1 0.86). Our method offers a rapid and cost-effective approach to assess the footprint of herbicide applications for improved weed management.

## Analysis Overview

This project implements a comprehensive machine learning pipeline for automated herbicide dye detection from drone imagery. The analysis follows a systematic approach that begins with hyperparameter optimization using Bayesian methods to identify optimal learning rates, batch sizes, and data augmentation strategies. A key innovation of this work is the development and evaluation of synthetic data generation techniques, where colored patches are overlaid on background images to augment the training dataset and improve model generalization.

The experimental design compares three distinct training approaches: models trained exclusively on real data, models trained only on synthetic data, and hybrid models that combine both real and synthetic training examples. This comparison reveals important insights about the effectiveness of synthetic data augmentation in computer vision tasks, particularly for specialized applications like herbicide dye detection where collecting large amounts of labeled training data can be challenging and expensive.

The analysis demonstrates that hybrid training approaches, which combine real field data with optimally parameterized synthetic examples, achieve superior performance compared to either approach alone. The systematic evaluation across different seasonal conditions, dye colors, application concentrations, and patch sizes provides comprehensive validation of the method's robustness and practical applicability for real-world weed management scenarios.

## Dependencies

### Core Machine Learning Libraries
- **PyTorch** (`torch`) - Deep learning framework for neural networks, tensors, and optimizers
- **torchvision** - Computer vision utilities for image transformations and data augmentation
- **transformers** (Hugging Face) - Vision transformer models (DINOv2) via torch.hub
- **datasets** (Hugging Face) - Dataset loading and processing for 'mpg-ranch/dye_test' dataset

### Data Processing
- **NumPy** (`numpy`) - Numerical computing and array operations
- **pandas** - Data manipulation, CSV loading, and result dataframe creation
- **PIL** (Python Imaging Library) - Image processing, cropping, and drawing operations

### Machine Learning Evaluation
- **scikit-learn** (`sklearn`) - F1 score calculations and model evaluation metrics

### Hyperparameter Optimization
- **Optuna** - Bayesian optimization framework for hyperparameter tuning
- **kaleido** - Static image export for Optuna visualization plots

### Visualization
- **matplotlib** (`matplotlib.pyplot`) - Plotting, visualization, and image display
- **seaborn** - Statistical data visualization with enhanced aesthetics

### Utilities
- **pickle** - Python object serialization for saving Optuna studies
- **json** - JSON data handling for experiment results
- **colorsys** - Color system conversions for synthetic data generation
- **os**, **random**, **functools**, **math** - Standard library utilities

### Google Colab Specific
- **google.colab** - Colab utilities for secrets management and runtime control

## Notebook Descriptions

### Data Optimization and Hyperparameter Tuning

**01_learning_hsweep.ipynb**: This notebook performs hyperparameter optimization for basic learning parameters using Optuna. It optimizes learning rate and batch size for a DinoV2 model trained on the dye detection dataset across three months (May, July, October). The main experiment uses 150 trials with early stopping to find optimal training parameters that maximize F1 scores for binary classification between blue/red dyes vs. no-dye background.

**02_synthetic_props_hsweep.ipynb**: This notebook optimizes synthetic data generation parameters for creating artificial dye patches. It uses the best learning parameters from notebook 01 and focuses on tuning synthetic dye properties including hue, value, saturation, and opacity ranges for both red and blue dyes. The goal is to find optimal parameters for generating synthetic training data that improves model performance on real dye detection.

**03_aug_hsweep.ipynb**: This notebook performs hyperparameter optimization for data augmentation strategies. It combines results from the previous notebooks and optimizes augmentation parameters including synthetic data probability, random cropping, flipping, rotation, and color jittering. The experiment determines the best combination of augmentations to improve model generalization while using both real and synthetic training data.

### Training Duration Optimization

**05_epoch_hsweep.ipynb**: This notebook determines the optimal number of training epochs using the best hyperparameters found in previous experiments. It trains models for up to 50 epochs using 5-fold cross-validation and tracks F1 scores to identify when performance peaks. The results show the optimal epoch count is 27 for the hybrid real+synthetic training approach.

**05b_epoch_hsweep_synthetic_only.ipynb**: This notebook is similar to 05_epoch_hsweep but trains exclusively on synthetic data (label 0 images with synthetic dye overlays). It determines that 8 epochs is optimal when training only on synthetically generated dye examples. The experiment helps understand how synthetic-only training compares to hybrid approaches.

**05c_epoch_hsweep_real_only.ipynb**: This notebook finds optimal epochs for training on real data only (no synthetic augmentation). It sets synthetic probability to 0 and trains for up to 50 epochs, determining that 25 epochs is optimal for real-only training. This provides a baseline comparison against synthetic and hybrid training approaches.

### Model Training and Saving

**06_save_test_model.ipynb**: This notebook trains and saves the final hybrid model using all optimized hyperparameters. It trains for the optimal 27 epochs using the best learning rate, batch size, augmentation parameters, and synthetic data generation settings. The trained model state dictionary is saved as 'test.pth' for later evaluation on test data.

**06b_save_test_model_synthetic_only.ipynb**: This notebook trains and saves a model using only synthetic data for 8 epochs. It filters the training dataset to only include label 0 (no-dye) images and applies synthetic dye overlays during training. The model is saved as 'test_synthetic_only.pth' to compare performance against hybrid and real-only approaches.

**06c_save_test_model_real_only.ipynb**: This notebook trains and saves a model using only real training data for 25 epochs. It sets synthetic probability to 0 to exclude any synthetic data generation and trains on the original dataset without artificial dye overlays. The model is saved as 'test_real_only.pth' for comparison studies.

### Performance Evaluation

**07_eval_test_performance.ipynb**: This notebook evaluates the hybrid model's performance on the test dataset. It loads the saved 'test.pth' model and runs inference on test images, computing F1 scores for blue and red dye detection. The results show a mean F1 score of 0.9506, with individual scores of 0.9222 for blue and 0.9789 for red dye detection.

**07b_eval_test_performance_synthetic_only.ipynb**: This notebook evaluates the synthetic-only trained model on test data. It loads 'test_synthetic_only.pth' and computes test performance metrics, achieving a lower mean F1 score of 0.9129 (0.8636 for blue, 0.9622 for red). This demonstrates that synthetic-only training underperforms compared to the hybrid approach.

**07c_eval_test_performance_real_only.ipynb**: This notebook evaluates the real-only trained model on test data. It loads 'test_real_only.pth' and achieves a mean F1 score of 0.9298 (0.9290 for blue, 0.9307 for red). The results show that real-only training performs better than synthetic-only but slightly worse than the hybrid approach.

### Visualization and Analysis

**08_plot_synthetic_v_hybrid.ipynb**: This notebook creates comparative visualizations of model performance across different training approaches. It generates bar plots comparing F1 scores for blue and red dye detection across synthetic-only, hybrid (real+synthetic), and real-only training methods. The plots demonstrate that the hybrid approach achieves the highest overall performance.

**09_plot_factor_level_f1.ipynb**: This notebook analyzes model performance across different experimental factors including month, dye color, concentration, and patch size. It creates detailed visualizations showing F1 scores broken down by these factors, revealing how performance varies across different dye concentrations (light vs. dark) and patch sizes (0.1m vs. 0.5m) across different months.

**10_plot_example_tiles.ipynb**: This notebook creates visual examples of the dye detection dataset by stitching together image tiles. It groups images by experimental conditions and creates composite images showing how dye patches appear across different months (May, July, October). The notebook generates example tiles for different color-concentration-size combinations to illustrate the dataset's visual characteristics.

**11_fake_dye_examples.ipynb**: This notebook demonstrates the synthetic dye generation process using the optimized parameters. It loads the best synthetic parameters from the hyperparameter optimization and applies the SuperimposeSquare transform to create artificial dye patches on background images. The notebook visualizes a grid of synthetic dye examples to show how the artificial training data appears.

**12_report_best_synthetic_params.ipynb**: This notebook extracts and reports the optimal synthetic dye parameters found through Bayesian optimization. It loads the best parameters from the synthetic properties study and formats them into a LaTeX table suitable for publication. The optimal parameters include specific hue, value, saturation, and opacity ranges for generating realistic synthetic red and blue dye patches.