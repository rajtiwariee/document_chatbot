#!/bin/bash

# Base URL
API_URL="http://localhost:8000/api"

echo "--------------------------------------------------"
echo "1. Registering User..."
echo "--------------------------------------------------"
curl -s -X POST "$API_URL/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123",
    "organization_name": "TestCorp",
    "full_name": "Test User"
  }' | json_pp

echo -e "\n\n--------------------------------------------------"
echo "2. Logging In..."
echo "--------------------------------------------------"
TOKEN_RESP=$(curl -s -X POST "$API_URL/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=password123")

ACCESS_TOKEN=$(echo $TOKEN_RESP | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
echo "Access Token: $ACCESS_TOKEN"

if [ -z "$ACCESS_TOKEN" ]; then
  echo "Login failed!"
  exit 1
fi

echo -e "\n\n--------------------------------------------------"
echo "4. Uploading Document..."
echo "--------------------------------------------------"
UPLOAD_RESP=$(curl -s -X POST "$API_URL/documents/upload" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "file=@test1.txt")
echo $UPLOAD_RESP | json_pp

DOC_ID=$(echo $UPLOAD_RESP | grep -o '"id":"[^"]*' | cut -d'"' -f4)
echo "Document ID: $DOC_ID"

echo -e "\n\n--------------------------------------------------"
echo "5. Checking Document Status..."
echo "--------------------------------------------------"
curl -s -X GET "$API_URL/documents/$DOC_ID" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | json_pp

echo -e "\n\n--------------------------------------------------"
echo "6. Listing Documents..."
echo "--------------------------------------------------"
curl -s -X GET "$API_URL/documents/" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | json_pp

echo -e "\n\n--------------------------------------------------"
echo "7. Chatting with Agent..."
echo "--------------------------------------------------"
# Wait a bit for processing (mock wait since we don't have real background worker success guaranteed here)
echo "Waiting 10 seconds for processing..."
sleep 10

curl -s -X POST "$API_URL/chat/" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What documents do I have?"
  }' | json_pp

echo -e "\n\nDone!"
