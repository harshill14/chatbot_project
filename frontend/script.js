async function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput })
    });

    const result = await response.json();
    const chatBox = document.getElementById('chat-box');
    chatBox.innerHTML += `<div>User: ${userInput}</div>`;
    chatBox.innerHTML += `<div>Bot: ${result.response}</div>`;
    document.getElementById('user-input').value = ''; // Clear input
}
