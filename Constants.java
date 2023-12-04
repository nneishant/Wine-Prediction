package org.njit.np763;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

public class Constants {
    public static final Logger LOGGER = LogManager.getLogger(Constants.class);

    public static final String DATA_PATH = "data/";
    public static final String TRAINING_DATASET = DATA_PATH + "TrainingDataset.csv";
    public static final String VALIDATION_DATASET = DATA_PATH + "ValidationDataset.csv";
    public static final String MODEL_PATH = DATA_PATH + "TrainingModel";
    public static final String TESTING_DATASET = DATA_PATH + "TestDataset.csv";

    public static final String APP_NAME = "Wine-quality-test";
}
