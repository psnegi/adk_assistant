"""YouTube video search and detailed summary tool with transcript analysis."""

from __future__ import annotations

import logging
import os
from typing import Optional
from datetime import datetime, timedelta, timezone

from google.adk.tools import FunctionTool
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from personal_assistant.tools.retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)

# For transcript extraction
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    TRANSCRIPT_AVAILABLE = True
except ImportError:
    TRANSCRIPT_AVAILABLE = False


@retry_with_backoff(retryable_exceptions=(HttpError, Exception))
def search_youtube_videos(
    query: str,
    max_results: int = 5,
    days_back: int = 30,
    order_by: str = "relevance",
    channel_id: str = "",
    min_duration_minutes: int = 0
) -> str:
    """Search YouTube for recent videos/podcasts and return basic information.
    
    Args:
        query: Search query (e.g., "AI news", "Lex Fridman", "python tutorial")
        max_results: Maximum number of videos to return (default 5, max 50)
        days_back: How many days back to search (default 30)
        order_by: Sort order - "relevance", "date", "viewCount", "rating" (default "relevance")
        channel_id: Optional channel ID to search within a specific channel
        min_duration_minutes: Minimum video duration in minutes (0=any, 4=short, 20=medium/long)
    
    Returns:
        List of videos with titles, channels, URLs, and view counts
    """
    # Use YouTube Data API v3
    # Check if we have API key set
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return (
            "YouTube API key not found. Please set YOUTUBE_API_KEY environment variable.\n"
            "Get an API key from: https://console.cloud.google.com/apis/credentials"
        )
    
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        
        # Calculate date for published after filter
        published_after = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat("T").replace("+00:00", "Z")
        
        # Determine video duration filter
        video_duration = "any"
        if min_duration_minutes >= 20:
            video_duration = "long"  # > 20 minutes
        elif min_duration_minutes >= 4:
            video_duration = "medium"  # 4-20 minutes
        elif min_duration_minutes > 0:
            video_duration = "short"  # < 4 minutes
        
        # Build search parameters
        search_params = {
            "q": query,
            "part": "id,snippet",
            "maxResults": min(max_results, 50),
            "type": "video",
            "order": order_by if order_by in ["relevance", "date", "viewCount", "rating"] else "relevance",
            "publishedAfter": published_after,
            "videoDuration": video_duration
        }
        
        # Add channel filter if specified
        if channel_id:
            search_params["channelId"] = channel_id
        
        # Search for videos
        search_response = youtube.search().list(**search_params).execute()
        
        if not search_response.get("items"):
            return f"No videos found for query: {query}"
        
        results = []
        video_ids = []
        
        for item in search_response["items"]:
            video_id = item["id"]["videoId"]
            video_ids.append(video_id)
            snippet = item["snippet"]
            
            results.append({
                "video_id": video_id,
                "title": snippet["title"],
                "channel": snippet["channelTitle"],
                "published": snippet["publishedAt"],
                "description": snippet["description"][:200] + "..." if len(snippet["description"]) > 200 else snippet["description"],
                "url": f"https://www.youtube.com/watch?v={video_id}"
            })
        
        # Get video statistics (views, likes, etc.)
        stats_response = youtube.videos().list(
            part="statistics,contentDetails",
            id=",".join(video_ids)
        ).execute()
        
        # Merge statistics with results
        for i, stats_item in enumerate(stats_response.get("items", [])):
            if i < len(results):
                stats = stats_item["statistics"]
                results[i]["views"] = int(stats.get("viewCount", 0))
                results[i]["likes"] = int(stats.get("likeCount", 0))
                results[i]["duration"] = stats_item["contentDetails"]["duration"]
        
        # Format output
        output = [f"🔍 Found {len(results)} videos for: {query}\n"]
        
        for idx, video in enumerate(results, 1):
            output.append(
                f"📹 {idx}. {video['title']}\n"
                f"   Channel: {video['channel']}\n"
                f"   Views: {video.get('views', 0):,} | Likes: {video.get('likes', 0):,}\n"
                f"   Published: {video['published'][:10]}\n"
                f"   Description: {video['description']}\n"
            )
        
        return "\n".join(output)
        
    except HttpError as e:
        return f"YouTube API error: {e}"
    except Exception as e:
        return f"Error searching YouTube: {e}"


def get_video_summary(
    video_url: str,
    extract_books: bool = True,
    extract_quotes: bool = True,
    extract_references: bool = True,
    extract_key_points: bool = True
) -> str:
    """Get detailed summary of a YouTube video including transcript analysis.
    
    Extracts and analyzes the video transcript to identify:
    - Books mentioned
    - Historical quotes or notable quotations
    - References to people, papers, or resources
    - Main points and key takeaways
    
    Args:
        video_url: YouTube video URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)
        extract_books: Whether to identify books mentioned (default True)
        extract_quotes: Whether to extract notable quotes (default True)
        extract_references: Whether to identify references/citations (default True)
        extract_key_points: Whether to summarize key points (default True)
    
    Returns:
        Detailed summary with requested elements
    """
    if not TRANSCRIPT_AVAILABLE:
        return (
            "youtube-transcript-api not installed. Please install it:\n"
            "Run: uv pip install youtube-transcript-api"
        )
    
    # Extract video ID from URL
    video_id = None
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[1].split("?")[0]
    else:
        return "Invalid YouTube URL. Please provide a valid URL like: https://www.youtube.com/watch?v=VIDEO_ID"
    
    try:
        # Initialize the API
        api = YouTubeTranscriptApi()
        
        # Try to get transcript - first list available transcripts
        transcript_list = None
        transcript_language = "unknown"
        
        try:
            # Get list of available transcripts
            available_transcripts_info = api.list(video_id)
            
            # Try to fetch transcript (defaults to English if available)
            transcript_list = api.fetch(video_id, languages=['en'])
            transcript_language = "English"
        except Exception as e:
            # If English not available, try to get any available transcript
            try:
                transcript_list = api.fetch(video_id)
                transcript_language = "Available language (auto-detected)"
            except Exception as fetch_error:
                raise Exception(f"No transcripts available: {fetch_error}")
        
        if not transcript_list:
            return (
                f"❌ No transcripts found for this video (Video ID: {video_id}).\n\n"
                "This could be because:\n"
                "- The video owner has disabled captions\n"
                "- The video is too new (transcripts not generated yet)\n"
                "- The video is a live stream or premiere\n"
                "- The video is private, age-restricted, or deleted\n\n"
                "Please try a different video with available captions."
            )
        
        # Combine all transcript segments
        full_transcript = " ".join([entry.text for entry in transcript_list])
        
        # Get video metadata using YouTube API
        api_key = os.getenv("YOUTUBE_API_KEY")
        video_info = ""
        
        if api_key:
            try:
                youtube = build("youtube", "v3", developerKey=api_key)
                video_response = youtube.videos().list(
                    part="snippet,statistics,contentDetails",
                    id=video_id
                ).execute()
                
                if video_response.get("items"):
                    item = video_response["items"][0]
                    snippet = item["snippet"]
                    stats = item["statistics"]
                    
                    video_info = (
                        f"📹 **Video Information**\n"
                        f"Title: {snippet['title']}\n"
                        f"Channel: {snippet['channelTitle']}\n"
                        f"Published: {snippet['publishedAt'][:10]}\n"
                        f"Views: {int(stats.get('viewCount', 0)):,}\n"
                        f"Likes: {int(stats.get('likeCount', 0)):,}\n"
                        f"Duration: {item['contentDetails']['duration']}\n\n"
                    )
            except Exception:
                pass  # If API call fails, continue without metadata
        
        # Truncate transcript if too long (for AI analysis)
        # Most models have context limits, so we'll take first ~8000 words
        words = full_transcript.split()
        if len(words) > 8000:
            full_transcript = " ".join(words[:8000]) + "\n\n[... transcript truncated for analysis ...]"
        
        # Build analysis prompt
        analysis_sections = []
        
        if extract_key_points:
            analysis_sections.append("- Main points and key takeaways (numbered list)")
        if extract_books:
            analysis_sections.append("- Books mentioned (title and author if mentioned)")
        if extract_quotes:
            analysis_sections.append("- Notable quotes or historical quotations (with attribution if available)")
        if extract_references:
            analysis_sections.append("- References to people, papers, studies, or resources mentioned")
        
        result = video_info + (
            f"📝 **Transcript Analysis**\n\n"
            f"**Transcript Language:** {transcript_language}\n"
            f"**Transcript Length:** {len(words):,} words\n\n"
            f"**Full Transcript:**\n"
            f"{full_transcript}\n\n"
            f"---\n\n"
            f"**Please analyze this transcript and extract:**\n"
            + "\n".join(analysis_sections) +
            "\n\nProvide a well-organized summary with clear sections for each requested element."
        )
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide more detailed error messages
        if "No transcripts available" in error_msg or "Could not retrieve" in error_msg:
            return (
                f"❌ No transcripts available for this video (Video ID: {video_id}).\n\n"
                f"Error details: {error_msg}\n\n"
                "Possible reasons:\n"
                "- The video owner has disabled captions/subtitles\n"
                "- The video is too new (transcripts not generated yet - try again in a few hours)\n"
                "- The video is a live stream or upcoming premiere\n"
                "- The video is private, age-restricted, members-only, or deleted\n"
                "- The video is a YouTube Short (transcripts often unavailable)\n\n"
                "Try searching for a different video with captions enabled."
            )
        elif "Video unavailable" in error_msg or "invalid" in error_msg.lower():
            return (
                f"❌ Video not found or unavailable (Video ID: {video_id}).\n\n"
                f"Error: {error_msg}\n\n"
                "Please check:\n"
                "- Is the video URL correct?\n"
                "- Is the video public and available in your region?\n"
                "- Has the video been deleted or made private?"
            )
        else:
            return (
                f"❌ Error getting video transcript (Video ID: {video_id}).\n\n"
                f"Error details: {error_msg}\n\n"
                "Please try:\n"
                "- Checking if the video has captions/subtitles enabled\n"
                "- Trying a different video\n"
                "- Verifying the video URL is correct"
            )


def check_video_transcripts(video_url: str) -> str:
    """Check what transcripts are available for a YouTube video.
    
    Useful for debugging why a video summary might not work.
    
    Args:
        video_url: YouTube video URL
    
    Returns:
        List of available transcript languages
    """
    if not TRANSCRIPT_AVAILABLE:
        return "youtube-transcript-api not installed."
    
    # Extract video ID
    video_id = None
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[1].split("?")[0]
    else:
        return "Invalid YouTube URL."
    
    try:
        api = YouTubeTranscriptApi()
        transcript_info = api.list(video_id)
        
        return (
            f"📋 **Available Transcripts for Video ID: {video_id}**\n\n"
            f"{transcript_info}"
        )
    except Exception as e:
        return f"Error checking transcripts: {e}"


# Create tool instances
search_youtube_tool = FunctionTool(search_youtube_videos)
youtube_summary_tool = FunctionTool(get_video_summary)
check_transcripts_tool = FunctionTool(check_video_transcripts)
