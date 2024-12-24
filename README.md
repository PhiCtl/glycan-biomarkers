[<img src="public/221006-logo-isospec-color.jpg" width="400" align="right">]()


&nbsp;  

# Machine Learning Internship Assignment  

&nbsp;  

Welcome to your internship assignment! This two-week challenge is designed to evaluate your data science and machine learning capabilities through two interconnected tasks that reflect our team's actual workflow.

Our will is that that after completion, you have rough understanding of the domain and a first taste of how your internship would unfold. 

We will introduce to you the domain of glycobiology and the tools we use to characterize glycans. You are not expected to be an expert in the domain and we will provide you with the necessary background. In any case we also provide you with complementary resources for further explanation.

Enjoy :) ! 

## Introduction: What are Glycans?

Glycans are complex sugar molecules that attach to proteins and lipids on cell surfaces and in blood, playing crucial roles in cellular communication, immune response, and disease progression. These molecules undergo significant changes during disease development, making them valuable biomarkers for early detection and monitoring of various conditions including cancer. Their presence in blood and structural diversity make glycans particularly attractive targets for diagnostic development, as changes in their abundance or composition can signal specific disease states before other symptoms appear.


## Task 01: Biomarker Discovery

In this first phase, you are provided with a task that is very similar to the typical data science tasks our team has to work on. Your role is to find biological differences in molecular expression specific to disease patients or healthy controls. Your skills should allow you to understand the data you are provided, clear out the noise and extract the signal we are after. In the typical workflow at ISOSPEC this means finding molecules of interest that we later on characterize with our proprietary breakthrough technology (CIRIS). 

### Background

#### Chromatographic Peaks
After data engineering team processed the results from LC-MS experiments of lung cancer patients cohort study run by our laboratory, you are looking to analyse the differential expression of glycans between classes of interest, namely disease and control. Molecules under study are cleaved glycans off of proteins running in blood samples. Experiment read out consists of list of Chromatographic peaks, that can be described with three main information points: Retention time (time took to migrate through LC column) indicative of the molecules conformation, Mass to charge ratio from the detector, indicative of mass composition and intensity of detection, absolute value indicative of abundance of the molecule. Chromatographic peaks height (intensity) and shape are important information to assess the quality of the signal and the relative abundance of the molecule. As such the peak area, integration of the curve in the time domain bound by peak start and end is the metric used to quantify presence.

Put simply:
• Retention time shows when molecule emerges from the column
• Mass to charge ratio indicates molecular composition
• Peak intensity represents molecule abundance
• Peak area is our quantitative measure

#### LC-MS Experiments
LC-MS experiments follow a strict framework to reduce the **variability of the analytical platform** in the data. This framework is composed of different blocks reproduced identically through multiple batches. A schematic description of a batch is shown below:

![Sample List Organization](sample-list.png)
*Figure 1: Schematic representation of a typical batch showing the order and organization of different sample types. Blanks and System Suitability samples are run first, followed by alternating patterns of disease samples, controls, and quality checks.*

In essence, a block contains different sample types or "class" run in a specific order:
- **Blank samples**: No biological information, used to assess contaminants. Consider the Zero, Solvant and Blank samples of the schema. 
- **System Suitability (SS)**: Contains exogenous standards for monitoring detector behavior. Consider the SSS and SS Conditionnig samples of the schema. 
- **Disease/Control samples**: Main samples of interest, in the schema shown as RM and SRM. 
- **Quality Checks (QC)**: Mixture of all samples to monitor signal consistency. Pooled QC in the schema. 

Do not worry about the technical details of the experiment, you must gather at this stage samples have different class and purposes within the experiment. 

Your input data is summarized in three files, all in `data/input` in csv format:
1. **Data Matrix** (NxM):
   - Rows: Sample IDs (N samples)
   - Columns: Feature IDs (M features)
   - Values: Peak areas

2. **Feature Metadata** (MxK):
   - Rows: Feature IDs (M features)
   - Columns: Feature descriptors (RT, m/z) (K descriptors)

3. **Acquisition List** (NxD):
   - Sample metadata including batch and run order (D descriptors)
   - Rows: Sample IDs (N samples)

4. **Exogenous Standards** (Sxk):
   - Rows: Exogenous standards IDs (S standards)
   - Columns: Feature descriptors (RT, m/z) (k descriptors)

### A - Exploratory Data Analysis

Your first task is to inspect the data, understand the dynamics at hand and perform necessary analysis to assess if the experiments is exploitable for biomarker discovery. As the analytical platform can introduce measurement bias, it's crucial to monitor both technical variability (from the instrument) and biological variability (true differences between samples).

Some metrics to consider:
• Coefficient of variation (CV): Captures variability in peak area across samples
• D-Ratio: Compares technical to biological variation

You can find more on these metrics in the ressources section.

To guide your analysis, consider these questions:
1. How many features have been detected, how is the distribution across mz and retention time, and across classes? 
2. How is the contamination in the experiment? 
3. Are the standards detected consistently across the experiment?
4. How is the distribution of intensities across the classes? Are there any trends with respect to classes, batches, run order? 
5. What differences are there between sample classes? Is it consistent across batches? 

You are encorraged to provide figures, summary statistics to motivate any insights you gather. Feel free to expand from these questions to interrogate other aspects of the data.

### B - Data Processing

As you go about finding discriminative signals, you'll need to process your features in order to ensure consistency in their detection and comparability between samples. For that, filtering of the features and of the samples investigated is key for meaningful discovery. Think of this as cleaning your signal - just as a radio needs tuning to pick up a clear broadcast, we need to tune our data to hear the biological signal clearly through the technical noise.
Qualitative features should follow at least these criteria:

- Limited variability across samples of the same class (<30% on the QC)
- Consistent support in detection (>=70%) on the same class
- Within mass range of interest (Glycans > 500 m/z)

Make sure to interrogate the samples as well - each batch contains 92 well plates, and an error can happen! Monitor the sample-to-sample intensity variations and check for possible transformations that could improve comparability. 

The output of this phase should be a list of features and samples you will leverage for doing biomarker discovery. You should provide justifications for their selection or the removal of others - every choice matters when we're looking for reliable biomarkers.

### C - Discriminatory Analysis 

In this final step your role is to showcase your skills for extracting meaningful patterns from our experimental data. You will leverage the curated list of features to find a subset of them that captures most biological differences between classes. Here, little guidance is provided as we expect motivating the discovery of a biomarker should be one of your strong suits. Keep in mind that statistical significance, predictive power and decision boundaries are important concepts when justifying such discovery. A sense of hierarchy should also be provided in the final list of targets.

## Task 02: Biomarker Embedding

In this second phase, your role is to provide interpretation power to your findings. Following the discovery of a biomarker, you are tasked with leveraging third party knowledge around the structure you identified to provide more insights on the origin of this glycan, other diseases it has been identified and proteins it is related to. 

The laboratory team has successfully identified your discovered molecules using CIRIS technology, providing glycan sequences and structural compositions in the `glycan_list.csv` file. Now comes the exciting challenge of placing these discoveries in the broader context of glycobiology.

You have access to the glycowork library, a comprehensive glycan dataset which contains, as described by the authors:

"Glycowork currently contains the following main datasets that are freely available to everyone:
• df_glycan: contains ~50,500 unique glycan sequences, including labels such as ~39,500 species associations, ~19,000 tissue associations, and ~2,500 disease associations
• glycan_binding: contains >580,000 protein-glycan binding interactions, from 1,465 unique glycan-binding proteins"

Your task is to create an embedding space for the glycan libraries that captures meaningful relationships between molecules such as sequence proximity, origin similarity, disease commonalities and proteins they interact with. You will then embed your discovered glycans and assess their closeness to other structures to draw conclusions as to their nature. The embedding space can be learned by leveraging the features provided in the glycan_list columns: 

- `glycan_sequence`: Glycan sequence
- `composition`: Composition of the glycan
- `tissue_sample`: Medium sample collected
- `tissue_species`: Species associated with the sample collected


You can validate your learned representation by assessing the closeness of the N-glycans in the new embedding space. The N-glycans list is provided in the N-Glycans.csv. Also a notebook loading glycowork and isolating the N-glycans is provided in the notebooks folder.

The expected output is the enriched list of glycans with information as well as evidence for the utility of the embedding space you created. The more new information you can gather on the structures, the better. Be creative in the approaches you consider - any choice should be justified and insights drawn motivated.

# Submission Process

### Repository Setup
1. Fork the assignment repository from this repository
2. Create a new branch for your work
3. Commit your changes regularly with clear, descriptive messages
4. When complete, create a pull request to the main repository

### Required Documents
Your submission should include:

1. All code and analysis files
2. A reflection.md document containing:
   • Time spent on each section (E.g 8-10 days Task 1, 4-6 days Task 2)
   • Perceived difficulty of different components
   • Discussion of what worked well
   • Challenges encountered and how you addressed them
   • Any feedback on the assignment structure

# Technical Requirements

## Environment Setup

We use uv for package management - a modern, fast Python package installer and resolver. The project.toml file in the repository specifies all required dependencies, ensuring everyone works with the same environment. The lock file contains the exact versions of the dependencies that were installed. After cloning the repository, simply run `uv sync` and uv will handle the rest, creating a clean, isolated environment for your work. More on uv can be found in the ressources section.

Alternatively, you can use the requirements.txt file to install the dependencies with any package manager of your choice.


### Project Organization

• Clear directory structure separating data, code, and results
• Comprehensive README.md explaining setup and workflow
• Well-documented code following PEP 8 guidelines
• Proper error handling and input validation

### Evaluation Criteria

• Analytical rigor and statistical soundness
• Biological insight and interpretation depth
• Code quality and reproducibility
• Clear communication of methods and results
• Creative problem-solving approach


### Resources

• [LC MS Experiment](https://pyopenms.readthedocs.io/en/latest/user_guide/background.html)
• [D-Ratio](https://pmc.ncbi.nlm.nih.gov/articles/PMC10222478/)
• [Coefficient of Variation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3695475/)
• [Glaycan Analysis](https://www.mdpi.com/2218-273X/13/4/605)
• [Glycowork](https://github.com/BojarLab/glycowork)
• [UV package manager](https://docs.astral.sh/uv/guides/projects/)
