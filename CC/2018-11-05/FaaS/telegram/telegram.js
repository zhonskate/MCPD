const TelegramBot = require('node-telegram-bot-api');

const token = '761344338:AAHw4U44WKSb9rd0xGe2oFl71xZvHfXHcB8';

// Create a bot that uses 'polling' to fetch new updates
const bot = new TelegramBot(token, {polling: true});

// Matches "/echo [whatever]"
bot.onText(/\/echo (.+)/, (msg, match) => {
  // 'msg' is the received Message from Telegram
  // 'match' is the result of executing the regexp above on the text content
  // of the message

  const chatId = msg.chat.id;
  const resp = match[1]; // the captured "whatever"

  // send back the matched "whatever" to the chat
  bot.sendMessage(chatId, resp);
});

bot.onText(/\/start/, function onStartText(msg) {
    const chatId = msg.chat.id;
    resp = "welcome to the faas bot!"
    bot.sendMessage(chatId, resp);
  });

  bot.onText(/\/help/, function onStartText(msg) {
    const chatId = msg.chat.id;
    resp = "help!"
    bot.sendMessage(chatId, resp);
  });

  bot.onText(/\/registerFunction/, function onRFText(msg) {
    const chatId = msg.chat.id;
    resp = "send the tar.gz of the function node module"
    bot.sendMessage(chatId, resp);
    console.log(msg);
    file = bot.getFile(file_id);

    //console.log(file);
  });

  bot.on('message', (msg) => {
    const chatId = msg.chat.id;
    resp = msg;
    console.log(msg);
    if(msg.document){
        fileId = msg.document.file_id;
        fileName = msg.document.file_name;
        bot.getFile(fileId).then(function(){
            
        });
        console.log(ee);
    }
  });