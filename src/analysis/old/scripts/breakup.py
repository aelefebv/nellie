import tifffile
import os
import ome_types


im_dir = r'D:\test_files\nelly\20230330-AELxES-U2OS_dmr_PERK-stress_granules'
im_name = 'deskewed-2023-03-30_18-22-26_000_20230330-AELxES-U2OS_dmr_PERK-stress_granules-w3-all.ome.tif'
im_path = os.path.join(im_dir, im_name)
print('Getting memmap...')
tif_stack = tifffile.memmap(im_path, mode='r')
ome_xml = tifffile.tiffcomment(im_path)
ome = ome_types.from_xml(ome_xml, parser='lxml')

n_frames = tif_stack.shape[0]

chunk_size = 20

new_ome_xml = None
print('Writing chunks...')
for i in range(0, n_frames, chunk_size):
    print(i)
    chunk_dir = os.path.join(im_dir, 'chunks')
    path_im = os.path.join(chunk_dir, f'chunk_{i//chunk_size:0>5}-{im_name}')
    chunk = tif_stack[i:i + chunk_size]
    if not os.path.exists(path_im):
        tifffile.imwrite(path_im, chunk)
    if new_ome_xml is None:
        ome.images[0].pixels.size_t = chunk_size
        ome.images[0].pixels.tiff_data_blocks[0].plane_count = chunk_size*ome.images[0].pixels.size_z
        new_ome_xml = ome_types.to_xml(ome)
    tifffile.tiffcomment(path_im, new_ome_xml)
    # new_name = f'chunk_{i//chunk_size:0>5}-{im_name}'
    # new_path_im = os.path.join(chunk_dir, new_name)
    # os.rename(path_im, new_path_im)