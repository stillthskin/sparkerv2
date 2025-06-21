document.addEventListener("DOMContentLoaded", function() {
    const trainModel = document.getElementById('trainModel');
    const trainFeedDiv = document.getElementById('trainFeedDiv');
    
    trainModel.addEventListener('click', function() {
        // Use Axios to send a POST request
        axios.post('/train_model/', {
            // Optionally send data here if needed
            data: 'some_data',  
        })
        .then(function (response) {
            // Handle success
            if (response.data.status === 'success') {
                trainFeedDiv.innerHTML = '<p>' + response.data.message + '</p>';
            } else {
                trainFeedDiv.innerHTML = '<p>Error: ' + response.data.message + '</p>';
            }
        })
        .catch(function (error) {
            // Handle error
            trainFeedDiv.innerHTML = '<p>Request failed: ' + error + '</p>';
        });
    });
});
