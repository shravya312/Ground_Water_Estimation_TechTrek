# Google Maps API Setup

This application includes location-based groundwater analysis using Google Maps. To enable this feature, you need to set up a Google Maps API key.

## Setup Instructions

### 1. Get a Google Maps API Key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the following APIs:
   - Maps JavaScript API
   - Places API (optional, for enhanced location features)
4. Go to "Credentials" and create a new API key
5. Restrict the API key to your domain for security

### 2. Configure the API Key

#### Environment Variable (Required)
1. Create a `.env` file in the frontend directory
2. Add your API key:
   ```
   VITE_GOOGLE_MAPS_API_KEY=your_actual_api_key_here
   ```

**Note**: The application uses dynamic loading of the Google Maps API, so the API key must be set via environment variable. The system will automatically load the Google Maps API when the Location Analysis feature is used.

### 3. Test the Integration

1. Start the development server: `npm run dev`
2. Click the "📍 Location Analysis" button in the chat interface
3. Allow location access when prompted
4. The map should load and show your current location

## Features

- **Interactive Map**: Satellite view with location markers
- **Current Location Detection**: Automatic geolocation with user permission
- **Location-Based Analysis**: Groundwater data analysis for specific coordinates
- **Responsive Design**: Works on desktop and mobile devices

## Security Notes

- Never commit your API key to version control
- Use environment variables for API keys
- Restrict your API key to specific domains/IPs
- Monitor API usage in Google Cloud Console

## Troubleshooting

### Map Not Loading
- Check if the API key is correctly set
- Verify that Maps JavaScript API is enabled
- Check browser console for error messages

### Location Not Working
- Ensure HTTPS is enabled (required for geolocation)
- Check browser permissions for location access
- Verify that the user has granted location permission

### API Quota Exceeded
- Check your Google Cloud Console for usage limits
- Consider implementing API key rotation
- Monitor usage patterns and optimize requests
