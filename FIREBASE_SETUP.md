# Firebase Setup for Chat History

## Firestore Security Rules

Copy and paste the following rules into your Firebase Console > Firestore Database > Rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Allow users to read/write their own chat data
    match /users/{userId}/chats/{document} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Allow users to read/write their own user document
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```

## Database Structure

The chat history is stored in the following structure:

```
users/
  {userId}/
    chats/
      {messageId}/
        - role: "user" | "assistant"
        - text: "message content"
        - timestamp: serverTimestamp()
        - conversationId: "conv_1234567890_abc123"
        - id: "client_generated_id"
    - uid: "user123"
    - email: "user@example.com"
    - displayName: "User Name"
    - lastChatActivity: serverTimestamp()
    - createdAt: serverTimestamp()
    - updatedAt: serverTimestamp()
```

## Features

### 1. **Conversation Grouping**
- Each conversation has a unique `conversationId`
- Messages are grouped by conversation for easy retrieval
- Users can click on any history item to see the complete conversation

### 2. **Clickable History**
- History sidebar shows all previous user prompts
- Clicking a prompt loads the complete conversation (question + response)
- Visual indicators show which conversation is currently active

### 3. **Real-time Saving**
- All messages are automatically saved to Firebase
- User messages and AI responses are both stored
- Error messages are also saved for debugging

### 4. **Security**
- Users can only access their own chat data
- Authentication required for all operations
- Proper Firestore security rules implemented

## How to Deploy

1. **Go to Firebase Console**
2. **Navigate to Firestore Database**
3. **Click on "Rules" tab**
4. **Replace the existing rules with the provided rules**
5. **Click "Publish"**

## Testing

1. **Start the development server**: `npm run dev`
2. **Sign in with Google**
3. **Send some messages**
4. **Click on history items to see complete conversations**
5. **Use "New Chat" to start fresh conversations**

## Troubleshooting

### If you get permission errors:
- Make sure the Firestore rules are properly deployed
- Check that the user is authenticated
- Verify the user ID matches the document path

### If history doesn't load:
- Check the browser console for errors
- Verify Firebase configuration
- Make sure the user document exists

### If messages don't save:
- Check network connectivity
- Verify Firebase project configuration
- Check browser console for errors
