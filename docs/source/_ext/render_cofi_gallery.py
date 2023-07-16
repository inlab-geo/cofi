import sys
import pathlib

def render_rst(_):
    # preperation
    current_path = pathlib.Path(__file__).resolve().parent
    cofi_gallery_tools_path = current_path.parent / "cofi-gallery" / "tools"
    gallery_output_path = current_path.parent / "gallery" / "generated"
    sys.path.append(str(cofi_gallery_tools_path))
    try:        # locally
        import collect_examples
        # collect examples and generate rst
        all_data = collect_examples.collect_all()
        all_images = collect_examples.load_all_images(all_data, gallery_output_path)
        readme_rst = collect_examples.read_readme(cofi_gallery_tools_path.parent / "README.rst")
        gallery_rst = collect_examples.generate_gallery_rst(all_data, all_images)
        collect_examples.write_index_rst(readme_rst, gallery_rst, gallery_output_path)
    except:     # don't generate again in readthedocs
        pass

def setup(app):
    app.connect("builder-inited", render_rst)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

