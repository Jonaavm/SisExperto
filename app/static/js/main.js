const uploadBtn = document.getElementById("uploadBtn");
const csvFile = document.getElementById("csvFile");
const uploadInfo = document.getElementById("uploadInfo");
const targetCol = document.getElementById("targetCol");
const trainBtn = document.getElementById("trainBtn");
const trainInfo = document.getElementById("trainInfo");
const classifyForm = document.getElementById("classifyForm");
const classifyBtn = document.getElementById("classifyBtn");
const classifyResult = document.getElementById("classifyResult");

// Subir CSV
uploadBtn.addEventListener("click", async () => {
    if (!csvFile.files.length) return alert("Selecciona un CSV");
    const fd = new FormData();
    fd.append("file", csvFile.files[0]);
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();

    if (data.error) return alert(data.error);

    uploadInfo.innerHTML = `
        <p>Dataset cargado: ${data.shape[0]} filas, ${data.shape[1]} columnas.</p>
    `;

    // Rellenar select de target column
    targetCol.innerHTML = "";
    data.columns.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col;
        opt.textContent = col;
        targetCol.appendChild(opt);
    });

    // Crear formulario dinámico para clasificar
    buildClassifyForm(data.columns);
});

// Cambiar entre opciones kfold / holdout
document.getElementById("validation").addEventListener("change", (e) => {
    const val = e.target.value;
    document.getElementById("kfoldOptions").style.display = val === "kfold" ? "block" : "none";
    document.getElementById("holdoutOptions").style.display = val === "holdout" ? "block" : "none";
});

// Entrenar modelo
// Entrenar modelo
trainBtn.addEventListener("click", async () => {
    const payload = {
        algorithm: document.getElementById("algorithm").value,
        target: targetCol.value,
        validation: document.getElementById("validation").value,
        k_folds: document.getElementById("kFolds").value,
        test_size: document.getElementById("testSize").value,
        knn_k: document.getElementById("knnK").value,
        max_depth: document.getElementById("maxDepth").value,
    };

    trainInfo.innerHTML = "Entrenando modelo...";

    const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (data.error) return alert(data.error);

    // Mostrar métricas
    if (data.average) {
        trainInfo.innerHTML = `
            <p>Accuracy promedio: ${(data.average.accuracy * 100).toFixed(2)}%</p>
            <p>Precision promedio: ${(data.average.precision * 100).toFixed(2)}%</p>
        `;
        drawMetricsChart(data.average);
    } else {
        trainInfo.innerHTML = `
            <p>Accuracy: ${(data.metrics.accuracy * 100).toFixed(2)}%</p>
            <p>Precision: ${(data.metrics.precision * 100).toFixed(2)}%</p>
            <p>Recall: ${(data.metrics.recall * 100).toFixed(2)}%</p>
            <p>F1: ${(data.metrics.f1 * 100).toFixed(2)}%</p>
        `;

        drawMetricsChart(data.metrics);

        // 👇 Mostrar matriz de confusión si existe
        if (data.metrics.confusion_matrix) {
            drawConfusionMatrix(data.metrics.confusion_matrix);
        }

        // 👇 Mostrar árbol si es ID3
        if (data.tree) {
            document.getElementById("treeOutput").textContent = data.tree;
        }
    }
});


// Clasificar nuevo ejemplo
classifyBtn.addEventListener("click", async () => {
    const inputs = classifyForm.querySelectorAll("input");
    const sample = {};
    inputs.forEach(input => sample[input.name] = input.value);

    const res = await fetch("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample }),
    });
    const data = await res.json();

    if (data.error) return alert(data.error);
    classifyResult.textContent = `Predicción: ${data.prediction}`;
});

// Construir formulario dinámico para clasificar
function buildClassifyForm(columns) {
    classifyForm.innerHTML = "";
    columns.forEach(col => {
        const div = document.createElement("div");
        div.innerHTML = `
            <label>${col}:</label>
            <input name="${col}" />
        `;
        classifyForm.appendChild(div);
    });
}

// Dibujar gráfico de métricas
function drawMetricsChart(metrics) {
    const ctx = document.getElementById("metricsChart").getContext("2d");
    new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Accuracy", "Precision", "Recall", "F1"],
            datasets: [{
                label: "Métricas",
                data: [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1],
                backgroundColor: ["#4CAF50", "#2196F3", "#FFC107", "#F44336"],
            }],
        },
        options: {
            scales: { y: { beginAtZero: true, max: 1 } },
        },
    });
}

// Dibujar matriz de confusión
function drawConfusionMatrix(matrix) {
    const container = document.getElementById("confMatrix");
    container.innerHTML = "<h3>Matriz de Confusión</h3>";

    let html = "<table border='1' cellpadding='5' cellspacing='0'>";
    matrix.forEach(row => {
        html += "<tr>";
        row.forEach(val => {
            html += `<td style="text-align:center; min-width:30px;">${val}</td>`;
        });
        html += "</tr>";
    });
    html += "</table>";

    container.innerHTML += html;
}
