function submitFeedback(feedback) {
    const message = document.getElementById("displayed-message").innerText;
    const section = document.getElementById("feedback-section");
    const prediction = section.dataset.prediction;
    const model = section.dataset.model;

    // 감사합니다 표시 애니메이션
    const thankYou = document.getElementById('thank-you');
    thankYou.classList.add('visible');

    setTimeout(() => thankYou.classList.remove('visible'), 2000);
    setTimeout(() => {
        section.style.display = 'none';
        thankYou.style.display = 'none';
    }, 2500);

    // 서버로 전송
    fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            message: message,
            feedback: feedback,
            prediction: prediction,
            model: model
        })
    })
    .then(response => response.json())
    .then(data => console.log(data));
}
