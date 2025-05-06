export enum BookmarkType {
  DOCTOR = 'DOCTOR',
  ARTICLE = 'ARTICLE',
  HEALTH_RECORD = 'HEALTH_RECORD',
  HOSPITAL = 'HOSPITAL'
}

export interface Bookmark {
  id: string;
  user_id: string;
  item_id: string;
  item_type: BookmarkType;
  created_at: string;
  // Optional fields for UI display
  title?: string;
  subtitle?: string;
  image_url?: string;
}

export interface BookmarkFilter {
  type?: BookmarkType;
  sort_by?: 'newest' | 'oldest' | 'alphabetical';
  page?: number;
  limit?: number;
}
