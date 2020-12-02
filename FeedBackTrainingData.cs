using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace FeedbackPrediction
{
    public class FeedBackTrainingData
    {
        

        // this is our input -> features
        [LoadColumn(0)]
        public string FeedBackText { get; set; }

        // this is our prediction what we wanna to get from model
        [LoadColumn(1), ColumnName("Label")]
        public bool IsGood { get; set; }
        
       
    }

    /// <summary>
    /// Class used after model training 
    /// </summary>

    public class FeedbackPrediction : FeedBackTrainingData
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }

    }


}
