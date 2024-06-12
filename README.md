# data_quality
Methods for estimating if one would be better off _not_ using some data to train a model.

Terminology:

* Task: The application of some statistical method that uses labelled instances to estimate
  model parameters of any kind.
* Task labels: Ground truth annotations regarding the _task_, not the quality of the instances.
* Data quality: A numerical valuation of whether including an instance to the training data
  will make the task more likely to fail, i.e., to produce a model that performs worse by comparison
  to the model produced without this instance.
* Quality estimator: Any method for estimating the data quality of an instance. An estimator should
  be aware of what the task is, and may use the prediction of the current model on the instance.
  An estimator must not have access to any posterior information, such as task labels or quality labels.
* Quality labels: Ground truth annotations that relate to data quality. They do not
  necessarily assign specific quality valuations to each instance, but there must be at least one
  quality valuation assignment that consistently maps to the labels.
* Quality estimation task: The application of some mathod that uses labels (both quality labels and,
  if needed by the method, task labels) to create a _quality estimator_.

Structure of the repository:

* `timeseries/`: Code for accessing EEG data in npz format.
* `images/`: Code for accessing images.
* `annotator_agreement/`: Code for accessing task labels created by multiple annoators, combined with
  a single ground thruth label. The former are used to extract quality labels under the assumption that
  high annotator agreement signifies cleaner instances; whether this assumption holds depends on the task,
  so this code may or may not be appropriate for extracting quality labels. Regardless of this, single
  ground truth label can also be used as task label.
