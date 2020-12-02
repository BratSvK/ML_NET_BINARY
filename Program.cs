using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using Microsoft.VisualBasic.CompilerServices;


namespace FeedbackPrediction
{
    // we need a binnary clasification
    class Program
    {
        // define our dataSet file, field to hold the recently downloaded dataset file path:
        private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "inputData.txt");

        static void Main(string[] args)
        {
            
            // Step 1: Create a mlContext object of everything what we need in ML
            MLContext mlContext = new MLContext();

            // Step 2: We need a training dataSet
            TrainTestData splitDataView = LoadData(mlContext);

            // Step 3 build and traing a model 

            ITransformer model = BuildAndTraingModel(mlContext, splitDataView.TrainSet);

            // Step 4 : Evalueate the model 

            Evaluate(mlContext,model,splitDataView.TestSet);
            string line = "";
            // step 5 : Test model with user input
            do
            {
                Console.WriteLine("Please enter a statement: ");
                line = Console.ReadLine();

                UseModelWithSingleItem(mlContext, model, line.Trim());

            } while (!line.Equals("Y")); 
            
            UseModelWithBatchItems(mlContext,model);
            
        }

        /// <summary>
        /// Step 1 : Loads the data,
        /// step 2: splits the loaded dataset into traing test sets,
        /// step 3: returns dataset for training 
        /// </summary>
        /// <param name="mlContext"></param>
        public static TrainTestData LoadData(MLContext mlContext)
        {
            //Step 1: we need a dataSet in correct format , musi byt viac ako 2kb
            IDataView dataView = mlContext.Data.LoadFromTextFile<FeedBackTrainingData>(_dataPath, separatorChar:',', hasHeader:false);
            
            // Step 2 -> split for testing and training sets, we got two sets training and testing 
            TrainTestData spliDataView = mlContext.Data.TrainTestSplit(dataView, 0.2);          // fraction is evaluate more data from dataSet, lepsie sa opytat na toto !!! 

            // step 3 : return a trainTestData
            return spliDataView;

        }

        /// <summary>
        /// Methods for creatiion a model a test it
        /// -> 1. Extracts and transforms the data.
        /// -> 2. Trains the model.
        /// -> 3. Predicts sentiment based on test data.
        /// -> 4. Returns the model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="splitTrainSet"></param>
        /// <returns></returns>
        public static ITransformer BuildAndTraingModel(MLContext mlContext, IDataView splitTrainSet)
        {
            // Step 1 : Extracts and transforms the data. transform data to machine learning language text to numeric key type Features
            // Step 2 : train our model we need binary clasification task

            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features",
                inputColumnName: nameof(FeedBackTrainingData.FeedBackText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName:"Label",featureColumnName:"Features"));

            // step 3 : train the model -> aky algoritmus pouzijem na model s datami 
            Console.WriteLine("================== Create and Traing the Model");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;

        }

        /// <summary>
        /// Teestovat nas model s nasimi testovacimi datami
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        /// <param name="splitTestSet"></param>
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {

            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);

            // Return how model is performning with our actual Labels in the test dataset 
            CalibratedBinaryClassificationMetrics metrics =
                mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");                          // aky mame presny model 
            Console.WriteLine("=============== End of model evaluation ===============");



        }

        /// <summary>
        /// Creates a single comment of test data.
        /// Predicts feedback based on test data.
        /// Combines test data and predictions for reporting.
        /// Displays the predicted results.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        public static void UseModelWithSingleItem(MLContext mlContext, ITransformer model, string input)
        {
            // perform prediction on single item we neeed API 
            PredictionEngine<FeedBackTrainingData, FeedbackPrediction> predictionFunction =
                mlContext.Model.CreatePredictionEngine<FeedBackTrainingData, FeedbackPrediction>(model);

            // creating out testing feedback
            FeedBackTrainingData sampleStatement = new FeedBackTrainingData()
            {
                FeedBackText = input
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Feedback: {resultPrediction.FeedBackText} | Prediction: {(Convert.ToBoolean(resultPrediction.IsGood) ? "Positive" : "Negative")} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();

        }


        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            // we need create list of statements 
            IEnumerable<FeedbackPrediction> feedbacks = new List<FeedbackPrediction>()
            {
                new FeedbackPrediction
                {
                    FeedBackText = "This was a bad meal"
                },
                new FeedbackPrediction
                {
                    FeedBackText = "This was good chicken"
                }
            };

            // spravit to na nas tvar IDataview
            IDataView batchComments = mlContext.Data.LoadFromEnumerable(feedbacks);
            // potom potrebujeme prediciu z toho 
            IDataView predictions = model.Transform(batchComments);
            // spravit hlavnu cast predictie 
            IEnumerable<FeedbackPrediction> predictionsResult =
                mlContext.Data.CreateEnumerable<FeedbackPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            foreach (FeedbackPrediction feedbackPrediction in predictionsResult)
            {
                Console.WriteLine($"Sentiment: {feedbackPrediction.FeedBackText} | Prediction: {(Convert.ToBoolean(feedbackPrediction.IsGood) ? "Positive" : "Negative")} ");
            }
            Console.WriteLine("=============== End of predictions ===============");

        

        }
        
        

    }
    
       
    
}
