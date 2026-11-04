# Vinyl Music Café（黑膠音樂咖啡館）: Answers questions about the café, takes song wishes and reserves tables.

An AI customer service agent for a small, made-up music café in Kaohsiung with many vinyl records and live music on Fridays. Guests can ask about the café rules, songs, the menu and free tables, wish for a song, get the total of an order, and reserve or cancel a table, in English or in Traditional Chinese. The agent answers only from the café's own database and rules text through tools, so it does not make up opening hours, prices, rules or free tables.

> Course midterm project: My Own Language Model App (tool calling and RAG), built on the Unit G customer service agent

> Student: C115451102 席喬安

![Main screen of my agent, with the tool trace](images/main-1.png)
![Main screen of my agent, with the tool trace](images/main-2.png)


## Who It Serves and When They Use It

- **Who uses it:** guests of the café (students, music lovers, café visitors), and the staff, who see the tables and reservations in a staff tab.
- **When they use it:** before their visit (rules, opening hours, free tables, reservations) and during their visit (song wishes, the menu, the price of an order, the live music fee).
- **What problem it solves:** the staff do not have to answer the same questions again and again at the counter or on the phone, and guests get a correct answer from the café's own data instead of a guess from a general chatbot. Reservations and song wishes are really saved.

## What It Can Do

| Function | Example question | Tool it uses |
|---|---|---|
| Tell café facts (hours, address, phone, payment, delivery, parking, Wi-Fi, how song wishes work, buying records) | "What time do you open?", "可以刷信用卡嗎？" | `get_cafe_info` |
| Answer questions about the café rules (reservations, cancellations, no-show fee, live music and its fee, house rules, allergens, complaints, lost and found) | "Do I have to pay for the live music?", "訂了位沒有到會怎麼樣？" | `search_cafe_rules` (RAG) |
| Find songs and singers | "Do you have songs by Huang Shu-Jun?", "有五月天的勇敢嗎？" | `search_songs` |
| Put a song wish into the queue (only after the guest says yes) | "I would like to hear 在水一方." then "Yes, please." | `search_songs`, `request_song` |
| List drinks, desserts and sandwiches and tell prices | "What desserts do you have?", "How much is a coffee?" | `search_menu`, `get_price` |
| Work out the total of an order, including the live music fee | "Three of us come on Friday at 20:00 and order three coffees. How much is that?" | `calculate_total` |
| Tell how many tables are free now, or if a table is free on a day | "Is a table free right now?", "Do you have a table for 3 people tomorrow at 19:00?" | `get_table_status`, `check_availability` |
| Reserve a table (only after the guest says yes) | "I would like to reserve a table for 3 people tomorrow at 19:00. My name is Anna Lin and my phone number is 0912345678." | `check_availability`, `reserve_table` |
| Cancel a reservation (only after the guest says yes) | "Please cancel my reservation. The number is 1 and my phone number is 0912345678." | `cancel_reservation` |
| Hand over to the staff (song not on the list, buying a record, more than 4 people, a cancellation less than 3 hours before arrival, complaint, manager) | "Can you play Bohemian Rhapsody by Queen?", "I need a table for 8 people on Saturday at 19:00." | `transfer_to_human` |

## How It Works

Every question goes to the agent first. The agent chooses a tool, the tool reads the database, searches the rules text or calculates in code, and the agent answers only from what the tool returns. The chat page shows the tool trace for every answer.

- **Techniques:** tool calling (12 tools) and RAG (the café rules)
- **Model:** gemini-3.5-flash-lite for the chat, and gemini-embedding-001 for the rules search
- **Chat screen:** Gradio (version 6), with a chat tab and a staff tab
- **Languages:** English and Traditional Chinese. The program decides the reply language, not the model.
- **Database (SQLite):** built from the CSV files every time the program starts.

  | Table | Built from | What it holds | Rows |
  |---|---|---|---|
  | `cafe_info` | `cafe_info.csv` | café facts such as opening hours, address, payment, parking (English and Chinese text) | 9 |
  | `menu` | `menu.csv` | drinks, desserts and sandwiches: names in English and Chinese, category, price in NTD, available or not | 12 |
  | `songs` | `songs.csv` | singer and title of every song, in English and Chinese, and whether it is available | 16 |
  | `cafe_tables` | `tables.csv` | the 12 tables with 2 or 4 seats, and whether a table is occupied | 12 |
  | `reservations` | (created empty at every start) | table reservations: guest, phone, people, time, status (booked or cancelled) | 0 at start |
  | `queue` | (created empty at every start) | the song wishes that wait to be played | 0 at start |

- **RAG (the café rules):** `cafe_rules.txt` is cut into 11 chunks, one for every `##` heading. An embedding model turns every chunk into a vector. For a question, the tool `search_cafe_rules` turns the question into a vector, compares it with all chunks (cosine similarity) and returns the best 2 that are above a minimum score. The agent answers only from these chunks and names the source.
- **Writing data:** `request_song`, `reserve_table` and `cancel_reservation` have a `confirmed` argument. Nothing is written until the guest has said yes to the exact details. The code checks the reservation rules itself: 1 to 4 people, at least 2 hours before arrival, up to 14 days ahead, and within the opening hours.
- **QR code and staff tab:** the page `?table=5` (the QR code of table 5) shows a button "I sit here" that marks the table as occupied. In the staff tab (protected by a PIN) the staff free the tables again, and see and cancel reservations.
- **Safe agent loop:** it stops after 6 model calls, and it runs only the tools below.

- **Tools:**

  | Tool | What it does |
  |---|---|
  | `get_cafe_info(topic)` | reads one topic from `cafe_info` |
  | `search_cafe_rules(question)` | RAG: finds the chunks of `cafe_rules.txt` that fit the question |
  | `search_songs(keyword)` | finds songs by singer or title, in English or Chinese |
  | `search_menu(keyword)` | finds menu items by keyword, or lists a group (drinks, food, all) |
  | `get_price(item_name)` | price and availability of one menu item |
  | `calculate_total(items, live_music_guests)` | price × quantity of every item, and the live music fee, calculated in code |
  | `request_song(song_id, confirmed)` | writes a song into `queue`, only when `confirmed` is true |
  | `check_availability(date, time, party_size)` | checks the rules and if a table is free (nothing is written) |
  | `get_table_status()` | how many tables and seats are free right now |
  | `reserve_table(date, time, party_size, guest_name, phone, confirmed)` | saves a reservation, only when `confirmed` is true |
  | `cancel_reservation(reservation_id, phone, confirmed)` | cancels a reservation (reservation number and phone number needed), only when `confirmed` is true |
  | `transfer_to_human()` | gives the staff phone number |

## Sample Conversations

A song wish that is put into the queue:

![A song wish that is put into the queue (part 1)](images/song-request-1.png)
![A song wish that is put into the queue (part 2)](images/song-request-2.png)

A question about the café rules (RAG) and an order with the live music fee, calculated by the code:

![The café rules and the live music fee](images/rules-and-fee-1.png)
![The café rules and the live music fee](images/rules-and-fee-2.png)

A table is reserved after the guest says yes, and the reservation is cancelled:

![Reserving a table](images/reserve.png)
![Cancelling a reservation](images/cancel.png)

A reservation that does not exist:

![The agent does not make things up](images/not-made-up.png)

Free tables and the QR code of a table:

![Free tables before and after a guest scans the QR code](images/tables.png)

The staff tab with the saved reservations:

![The staff tab](images/staff-tab-1.png)
![The staff tab](images/staff-tab-2.png)
![The staff tab](images/staff-tab-3.png)


## Project Files

```text
music-cafe-agent/
├── app-gradio.py       # the agent: database, tools, rules search, agent loop and chat screen
├── cafe_info.csv       # café facts
├── menu.csv            # drinks, desserts and sandwiches
├── songs.csv           # the songs
├── tables.csv          # the 12 tables
├── cafe_rules.txt      # the café rules (the text for RAG)
├── tool_test.py        # the tool-choice test (27 questions), the rules search check and the confirmation tests
├── requirements.txt    # the Python packages to install
├── .env.example        # a sample settings file (no real key)
├── .gitignore          # files git must not upload: .env and *.db
├── images/             # screenshots for this README
└── README.md           # this file
```

## How to Install and Run It

1. Download the project:

   ```bash
   git clone https://github.com/JohannaSiegl/music-cafe-agent-v2.git
   cd music-cafe-agent-v2
   ```

2. Install the packages (Python 3.10 or newer):

   ```bash
   pip install -r requirements.txt
   ```

3. Make your settings file. Copy `.env.example` to `.env`, then fill in your own values.
   The chat model must support tool calling.

   ```bash
   copy .env.example .env      # Windows
   cp .env.example .env        # macOS or Linux
   ```

   ```text
   API_KEY=your key
   LLM_API_URL=the address of the chat server
   MODEL_NAME=the chat model
   EMBEDDING_API_KEY=         (empty: the chat server is used)
   EMBEDDING_API_URL=         (empty: the chat server is used)
   EMBEDDING_MODEL_NAME=the embedding model
   STAFF_PIN=a PIN for the staff tab
   ```

4. Start the agent. It builds the database from the CSV files, builds the rules index, and then starts the chat page:

   ```bash
   python app-gradio.py
   ```

5. Open the link shown in the console, usually <http://127.0.0.1:7860>. To try the QR code of a table, open <http://127.0.0.1:7860/?table=5>. The staff tab needs the PIN from `.env`.

To run the tests, stop the app first, then run:

```bash
python tool_test.py
```

## About the Data (Real and Made-Up)

- The **singers and songs** in the song list really exist. Only the **English song titles are my own translations** and might not be correct. The original titles are in Chinese.
- **Everything else about the café is made up and does not exist in real life.** This includes the café, its address, phone number, opening hours, tables, prices, menu, rules, the live music and the five performers. Names and phone numbers in the sample reservations are made up too.

## Notes

- The `.env` file holds the API key and the staff PIN. It is **not** in this repository. Make your own from `.env.example`.
- The agent does not play music. It only takes song wishes into the queue. Reservations, the queue and the occupied tables are empty at every start, because the database is built again from the CSV files.
- The tables are reserved for 2 hours and the agent gives the smallest table that fits. These numbers, the opening hours and the fees are written in `cafe_rules.txt` and in the constants of `app-gradio.py`. If you change one, change both.
- The agent cannot reserve for more than 4 people, and it cannot cancel less than 3 hours before the arrival. It hands these requests over to the staff.
- The minimum score of the rules search (`MIN_SCORE`) was chosen for the embedding model I used. Another embedding model may need another value (`python tool_test.py` shows the scores).
- On a free Gemini plan the number of calls per minute is limited, so the agent or the test script can get error 429 ("too many calls"). Wait a minute and try again.
