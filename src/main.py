
import os, zipfile
import supervisely as sly
import sly_globals as g
import gdown
from supervisely.io.fs import get_file_name
from supervisely.imaging.image import read
from cv2 import connectedComponents


def get_image_shape(img_path):
    im = read(img_path)

    return im.shape[0], im.shape[1]


def create_ann(img_path):
    labels = []

    height, width = get_image_shape(img_path)

    ann_name = get_file_name(img_path) + g.ann_suffix
    ann_path = os.path.join(g.annotations_path, ann_name)

    ann_mask = read(ann_path)[:, :, 0]

    for class_name, class_index in g.index_to_class.items():
        bool_mask = ann_mask == class_index
        ret, curr_mask = connectedComponents(bool_mask.astype('uint8'), connectivity=8)
        for i in range(1, ret):
            obj_mask = curr_mask == i
            if obj_mask.sum() < g.max_label_area:
                continue
            bitmap = sly.Bitmap(obj_mask)
            label = sly.Label(bitmap, g.meta.get_obj_class(class_name))
            labels.append(label)

    return sly.Annotation(img_size=(height, width), labels=labels)


def extract_zip():
    if zipfile.is_zipfile(g.archive_path):
        with zipfile.ZipFile(g.archive_path, 'r') as archive:
            archive.extractall(g.work_dir_path)
    else:
        g.logger.warn('Archive cannot be unpacked {}'.format(g.arch_name))
        g.my_app.stop()


@g.my_app.callback("import_weed")
@sly.timeit
def import_weed(api: sly.Api, task_id, context, state, app_logger):

    gdown.download(g.weed_url, g.archive_path, quiet=False)
    extract_zip()

    images_path = os.path.join(g.work_dir_path, g.folder_name, g.images_folder_name)
    g.annotations_path = os.path.join(g.work_dir_path, g.folder_name, g.annotation_folder_name)

    images_names = os.listdir(images_path)

    new_project = api.project.create(g.WORKSPACE_ID, g.project_name, change_name_if_conflict=True)
    api.project.update_meta(new_project.id, g.meta.to_json())

    new_dataset = api.dataset.create(new_project.id, g.dataset_name, change_name_if_conflict=True)

    progress = sly.Progress('Upload items', len(images_names), app_logger)
    for img_batch in sly.batched(images_names, batch_size=g.batch_size):

        img_pathes = [os.path.join(images_path, name) for name in img_batch]
        img_infos = api.image.upload_paths(new_dataset.id, img_batch, img_pathes)
        img_ids = [im_info.id for im_info in img_infos]

        anns = [create_ann(img_path) for img_path in img_pathes]
        api.annotation.upload_anns(img_ids, anns)

        progress.iters_done_report(len(img_batch))

    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "TEAM_ID": g.TEAM_ID,
        "WORKSPACE_ID": g.WORKSPACE_ID
    })
    g.my_app.run(initial_events=[{"command": "import_weed"}])


if __name__ == '__main__':
    sly.main_wrapper("main", main)
