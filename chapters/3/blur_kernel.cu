#define BLUR_SIZE 1 // The "radius" around the center pixel. This is for a 3x3 patch 

__global__ void blur_kernel(unsigned char* in, unsigned char* out, int w, int h)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (col < w && row < h)
  {
    int pix_val = 0;
    int pixels = 0;

    for (int blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE+1; ++blur_row)
    {
      for (int blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE+1; ++blur_col)
      {
        int cur_row = row + blur_row;
        int cur_col = col + blur_col;

        if (cur_row >= 0 && cur_row < h && cur_col >= 0 && cur_col < w)
        {
          pix_val += in[cur_row*w + cur_col];
          ++pixels;
        }
      }
    }
    out[row*w + col] = (unsigned char)(pix_val/pixels);
  }
}
