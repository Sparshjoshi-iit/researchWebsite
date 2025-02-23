css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://imgs.search.brave.com/Xtosz1KfYcbhhUnFaelf0JvfPVnya4SMLhQcCOhorz8/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWcu/ZnJlZXBpay5jb20v/ZnJlZS1waG90by9h/aS1nZW5lcmF0ZWQt/Y29uY2VwdC1odW1h/bl8yMy0yMTUwNjg4/Mzc1LmpwZz9zZW10/PWFpc19oeWJyaWQ" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://imgs.search.brave.com/kYYeYcSxp1O3JPxgGXa9lUN4rn_2rX7ZkO-fnScMP8I/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly90NC5m/dGNkbi5uZXQvanBn/LzA2LzcxLzE4LzYz/LzM2MF9GXzY3MTE4/NjM3N195ZFRQbkhk/VnExNWNCQUlLaEQx/aFpqWE92Rm44UXlQ/Ri5qcGc">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''