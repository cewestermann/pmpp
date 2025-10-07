#define CHANNELS 3

__global__ void color_to_grayscale(unsigned char* Pout, unsigned char* Pin, int width, int height)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;

  if (col < width && row < height) 
  {
    int gray_offset = row * width + col; // The offset into the linearized image
    int rgb_offset = gray_offset * CHANNELS; // The color image has 3 channels (R, G, B)  
    unsigned char r = Pin[rgb_offset    ]; // Red, the byte (0 - 255)
    unsigned char g = Pin[rgb_offset + 1]; // Green
    unsigned char b = Pin[rgb_offset + 2]; // Blue

    Pout[gray_offset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}
