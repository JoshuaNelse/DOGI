mkdir -p ~/.streamlit/

echo "[general]\n email = \"jshav28@wgu.edu\"\n " > ~/.streamlit/credentials.toml

echo " [server]\n headless = true\n port = $PORT\n " > ~/.streamlit/config.toml