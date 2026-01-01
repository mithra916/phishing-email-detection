function checkEmail() {
    const emailText = document.getElementById("emailText").value;

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ email_text: emailText })
    })
    .then(res => res.json())
    .then(data => {
        let resultDiv = document.getElementById("result");
        if (data.prediction === "Phishing Email") {
            resultDiv.style.color = "red";
        } else {
            resultDiv.style.color = "green";
        }

        resultDiv.innerHTML =
            `Prediction: ${data.prediction}<br>
             Confidence: ${data.confidence ? (data.confidence*100).toFixed(2) + '%' : 'N/A'}`;
    });
}
