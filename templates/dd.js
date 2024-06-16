$.ajax({
    url: '/predict',
    method: 'POST',
    data: formData,
    success: function(response) {
      var predictions = response.predictions;
      var resultDiv = $('#result');
      resultDiv.empty();
  
      for (var i = 0; i < predictions.length; i++) {
        resultDiv.append('<p>Prediction ' + (i + 1) + ': ' + predictions[i] + '</p>');
      }
  
      // Display the prediction chart
      if (response.actual_prices && response.dates) {
        displayPredictionChart(predictions, response.actual_prices, response.dates);
      } else {
        console.error('Missing data for chart');
      }
    },
    error: function(xhr, status, error) {
      alert('Error: ' + error);
    }
  });