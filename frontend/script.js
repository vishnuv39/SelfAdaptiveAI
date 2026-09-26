const API_URL = "http://127.0.0.1:8000";

let lastPrediction = null;


// --------------------------------
// Load active model
// --------------------------------

async function loadActiveModel() {

    try {

        const response = await fetch(`${API_URL}/model`);

        const data = await response.json();

        document.getElementById("activeModel").textContent =
            data.active_model;

    } catch (error) {

        document.getElementById("activeModel").textContent =
            "Unavailable";

        console.error(error);
    }
}


// --------------------------------
// Predict machine status
// --------------------------------

async function predictMachine() {

    const data = {

        Type: document.getElementById("type").value,

        air_temperature:
            Number(document.getElementById("airTemperature").value),

        process_temperature:
            Number(document.getElementById("processTemperature").value),

        rotational_speed:
            Number(document.getElementById("rotationalSpeed").value),

        torque:
            Number(document.getElementById("torque").value),

        tool_wear:
            Number(document.getElementById("toolWear").value)
    };


    const resultBox =
        document.getElementById("predictionResult");

    resultBox.innerHTML = "<p>Predicting...</p>";


    try {

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)
            }
        );


        const result = await response.json();


        if (!response.ok) {
            throw new Error(result.detail || "Prediction failed");
        }


        lastPrediction = {
            ...data,
            prediction:
                result.machine_status === "FAILURE" ? 1 : 0,

            model_version:
                result.model_version
        };


        resultBox.innerHTML = `

            <div class="result-status">
                ${result.machine_status}
            </div>

            <div class="result-probability">
                Failure Probability:
                <strong>
                    ${(result.failure_probability * 100).toFixed(2)}%
                </strong>
            </div>

            <div class="result-model">
                Model Used:
                <strong>${result.model_version}</strong>
            </div>

        `;


        loadActiveModel();

    } catch (error) {

        resultBox.innerHTML = `
            <p style="color:red;">
                Error: ${error.message}
            </p>
        `;

        console.error(error);
    }
}


// --------------------------------
// Submit actual feedback
// --------------------------------

async function submitFeedback(actual) {

    if (!lastPrediction) {

        document.getElementById("feedbackMessage").textContent =
            "Please make a prediction first.";

        return;
    }


    const feedbackData = {

        Type: lastPrediction.Type,

        air_temperature:
            lastPrediction.air_temperature,

        process_temperature:
            lastPrediction.process_temperature,

        rotational_speed:
            lastPrediction.rotational_speed,

        torque:
            lastPrediction.torque,

        tool_wear:
            lastPrediction.tool_wear,

        prediction:
            lastPrediction.prediction,

        actual:
            actual,

        model_version:
            lastPrediction.model_version
    };


    try {

        const response = await fetch(
            `${API_URL}/feedback`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(feedbackData)
            }
        );


        const result = await response.json();


        if (!response.ok) {
            throw new Error(result.detail || "Feedback failed");
        }


        document.getElementById("feedbackMessage").textContent =
            "✓ Feedback recorded successfully.";

    } catch (error) {

        document.getElementById("feedbackMessage").textContent =
            `Error: ${error.message}`;

        console.error(error);
    }
}

// --------------------------------
// Load monitoring data
// --------------------------------

async function loadMonitoring() {

    try {

        const response = await fetch(`${API_URL}/monitor`);

        const data = await response.json();

        document.getElementById("monitorModel").textContent =
            data.model_version;

        document.getElementById("monitorAccuracy").textContent =
            (data.accuracy * 100).toFixed(2) + "%";

        document.getElementById("monitorPrecision").textContent =
            (data.precision * 100).toFixed(2) + "%";

        document.getElementById("monitorRecall").textContent =
            (data.recall * 100).toFixed(2) + "%";

        document.getElementById("monitorF1").textContent =
            (data.f1_score * 100).toFixed(2) + "%";

        document.getElementById("monitorHealth").textContent =
            data.health;

        document.getElementById("monitorRetraining").textContent =
            data.retraining;

    } catch (error) {

        console.error(
            "Monitoring error:",
            error
        );

    }
}

async function simulateDegradation() {
    const button = document.getElementById("degradationButton");
    const message = document.getElementById("degradationMessage");

    button.disabled = true;
    button.textContent = "Retraining in progress...";
    message.textContent = "⚠ Simulating performance degradation...";

    try {
        const response = await fetch(`${API_URL}/simulate-degradation`, {
            method: "POST"
        });

        const data = await response.json();

        if (data.success) {
    message.textContent =
        "✓ Degradation detected and automatic retraining completed.";

    await loadActiveModel();
    await loadMonitoring();

} else {

    console.error("Backend error:", data);

    message.textContent =
        "✕ " + data.message;

    if (data.output) {
        console.error("Output:", data.output);
    }

    if (data.controller_output) {
        console.error(
            "Controller Output:",
            data.controller_output
        );
    }
}

    } catch (error) {
        console.error("Degradation simulation error:", error);
        message.textContent =
            "✕ Unable to run degradation simulation.";
    }

    button.disabled = false;
    button.textContent = "⚠ Simulate Model Degradation";
}


// Load monitoring data when page opens

loadMonitoring();

// --------------------------------
// Load model when page opens
// --------------------------------

loadActiveModel();