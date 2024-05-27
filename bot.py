import telebot 
import os 
from telebot.types import Message 
import asyncio 
from telebot.async_telebot import AsyncTeleBot 
from getTex import getTex 
from PIL import Image 
import re 
TOKEN = '7262268963:AAGVYgBIoD0yS3RUv00un6epGFUl9_JUHhM'
bot = AsyncTeleBot(TOKEN) 
def escape_markdown_v2(text): 
    special_characters = r'\_*[]()~`>#+-=|{}.!' 
    return re.sub(f'([{re.escape(special_characters)}])', r'\\\1', text) 
 
@bot.message_handler(commands=['start']) 
async def send_welcome(message: Message): 
    welcome_text = ( 
        "Hello! Welcome to the pic2tex bot.\n\n" 
        "This bot allows you to upload documents and photos, " 
        "and it will get LaTeX code from them to you. Here's how it works:\n\n" 
        "1. Send a document or photo to this chat.\n" 
        "2. The file will be processed, and LaTeX code will be extracted from it.\n" 
        "3. You will receive the LaTeX code in this chat.\n\n" 
        "Feel free to upload a document or photo to get started!" 
    ) 
    await bot.reply_to(message, welcome_text) 
 
async def handle_docs_photos(message: Message): 
    try: 
        folder_path = os.path.join(os.getcwd(), 'source') 
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path) 
 
        # Download file 
        file_id = message.document.file_id if message.document else message.photo[-1].file_id 
        file_info = await bot.get_file(file_id) 
        downloaded_file = await bot.download_file(file_info.file_path) 
 
        # Save file 
        file_name = message.document.file_name if message.document else f"image_{file_id}.jpg" 
        file_path = os.path.join(folder_path, file_name) 
 
        with open(file_path, 'wb') as new_file: 
            new_file.write(downloaded_file) 
 
        await bot.reply_to(message, f"Your file is being processed, please wait.") 
 
        # Call the getTex function (ensure that process_image is correctly used) 
        text = await asyncio.get_event_loop().run_in_executor( 
            None, getTex().process_image, Image.open(file_path) 
        ) 
 
         
        await bot.send_message(message.chat.id, f'''``` 
        {escape_markdown_v2(text)} 
        ```''', parse_mode='MarkdownV2') 
 
    except Exception as e: 
        await bot.reply_to(message, f"An error occurred: {e}") 
 
@bot.message_handler(content_types=['document', 'photo']) 
async def async_handle_docs_photos(message: Message): 
    await handle_docs_photos(message) 
 
async def main(): 
    await bot.polling() 
 
if __name__ == "__main__": 
    asyncio.run(main())