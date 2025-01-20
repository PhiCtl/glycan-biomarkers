# Isospec analytics interview project

The project directory is structured as follows:

```python
/isospec-internship/
├── data/
├── models/ # For Task 2
├── notebooks/ # For analysis
│   ├── data_processing_&_classification.ipynb
│   ├── eda.ipynb
│   └── glycan_embedding.ipynb
├── scripts/
│   ├── classification.py
│   └── data_processing.py
├── public/
├── README.md
├── project_outline.md # As initially provided
├── reflection.md
├── uv.lock
└── requirements.txt
```

## Task 1

### EDA

* The dataset contains 124 samples with 256 extracted features. The features considered are the chromatographic peaks detected for a given retention time range at a given mass / charge ratio range.  A feature value is the peak intensity integrated over the considered time range. Features intensity ranges on 5 order of magnitude, with 90% of features having a median intensity between 100 and 10'000. To improve comparability, we will further consider intensities on the log-scale.
* Blank samples batch analysis strongly suggests there is no contamination and a study on system suitability samples validates standard samples detection in the experiment.

* **Coefficient of Variation on QC samples**. The relative standard deviation or Coefficient of Variation measures the dispersion of QC measurements, thus repeatability. 50% of features have a coefficient of variation lower than 20% on QC samples, and 75% have a CV below 30%.
* **D-ratio on QC samples**. The D-ratio monitors how much of the total observed variation is due to technical variation. Using median average deviation estimator to compute the D-ratio, we find that 33 features have a D-ratio below 50%.

More on exploratory data analysis can be found in ```notebooks/eda.ipynb```.

### Data processing

Data processing is detailed in ```notebooks/data_processing_&_classification.ipynb``` and can be run independently with ```scripts/data_processing.py```.
We start with 252 features and 79 samples, we end up with 175 features and 67 samples. The data processing pipeline is the following:

* Feature selection :
  * Feature CV on QC samples < 30%
  * Consistent support in detection : select features for which the intensity is above 500 for at least 70% of the samples of the same class
  * D-ratio (optional) : select all features for which D-ratio is below 50%
* Outliers removal : remove samples for which there is at least 4 outliers, based on the Inter Quartile range criterion.
* Feature intensity transformation : for better comparison, map intensities to log scale.

#### Left to do

- [ ] implement intra batch effect correction
- [ ] add figures to README

I would have liked to correct intensities with QC samples in order to remove intra-batch effect, using spline interpolation. The method is described in the paper [Evaluation and correction of injection order effects in LC-MS/MS based targeted metabolomics](https://www.sciencedirect.com/science/article/pii/S1570023222004184#s0010) and implemented by [tidyms](https://tidyms.readthedocs.io/en/latest/).

### Discriminatory analysis

Discriminatory analysis is detailed in ```notebooks/data_processing_&_classification.ipynb``` and can be run independently with ```scripts/classification.py```.

To perform feature ranking, several approaches can be considered. I started with a Random Forest classifier, because it is more robust to overfitting than Decision Trees and because it enables feature importance ranking, based on their gini index.
I quickly perform cross validation on training samples to search the best number of tree estimators, fixing the `max_depth` to 3 to prevent over fitting since we have more features than samples. The best hyper parameter for the number of tree estimators is 20.

#### Results

| classes  | precision| recall   | f1-score | support  |
|----------|----------|----------|----------|----------|
| Dunn     | 0.86     | 1.00     | 0.92     | 6        |
| French   | 1.00     | 0.80     | 0.89     | 5        |
| LMU      | 1.00     | 1.00     | 1.00     | 3        |

| metrics  |          |          |          | support  |
|----------|----------|----------|----------|----------|
| accuracy |          |          |     0.93 | 14       |
| macro avg| 0.95     | 0.93     | 0.94     | 14       |
| w. avg   | 0.94     | 0.93     | 0.93     | 14       |

The binary accuracy for the classification task is of 0.93, since one sample of French has been misclassified as Dunn.

Using feature importance from Random Forest classifier, we end up with features 43, 44, 16, 46, 143, 53, 176, 24, 27 and 20 as top-10 features for the classifier to make a decision.
After running a Student t-test, we can assess at the 95% confidence level for features 16, 53 and 176, intensities mean are distinct between Lung cancer and Healthy groups and between Lung cancer and benign disease groups while the difference is not statistically significant between the benign disease and the healthy group.
For all the features that are left, feature mean intensity is different across the three classes.

#### Left to do

* [ ] use Shapley values for model decision explanation
* [ ] define random forest decision boundaries
* [ ] try out classification using MLP
* [ ] use Shapley values for model decision explanation
* [ ] define random forest decision boundaries
* [ ] refine error handling and input validation
* [ ] add function comment

## Task 2

### Left todos

* [ ] add figures to README
* [ ] validate neighbours finding method
* [ ] try out embedding learning with graph transformers
* [ ] highlight properly evidences for embedding usefulness
* [ ] find statistical significance of disease association for embedded glycans of interest

## Ressources

* [LC MS Experiment](https://pyopenms.readthedocs.io/en/latest/user_guide/background.html)
* [D-Ratio](https://pmc.ncbi.nlm.nih.gov/articles/PMC10222478/)
* [Coefficient of Variation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3695475/)
* [Guidelines for the use of QC samples](https://link.springer.com/article/10.1007/s11306-018-1367-3)
* [LC-MS data processing with python](https://pmc.ncbi.nlm.nih.gov/articles/PMC7602939)
* [Glycan Analysis](https://www.mdpi.com/2218-273X/13/4/605)
* [Glycowork](https://github.com/BojarLab/glycowork)
* [Using graph convolutional neural networks to learn a representation for glycans](https://www.sciencedirect.com/science/article/pii/S2211124721006161#sec1)
* [Graphormer from HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/graphormer#graphormer)
* [Graph transformers](https://huggingface.co/blog/intro-graphml)
* [UV package manager](https://docs.astral.sh/uv/guides/projects/)