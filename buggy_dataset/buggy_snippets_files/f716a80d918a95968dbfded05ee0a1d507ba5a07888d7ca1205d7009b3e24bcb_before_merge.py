def get_container_data_volumes(container, volumes_option, tmpfs_option, mounts_option):
    """
        Find the container data volumes that are in `volumes_option`, and return
        a mapping of volume bindings for those volumes.
        Anonymous volume mounts are updated in place instead.
    """

    volumes = []
    volumes_option = volumes_option or []

    container_mounts = dict(
        (mount['Destination'], mount)
        for mount in container.get('Mounts') or {}
    )

    image_volumes = [
        VolumeSpec.parse(volume)
        for volume in
        container.image_config['ContainerConfig'].get('Volumes') or {}
    ]

    for volume in set(volumes_option + image_volumes):
        # No need to preserve host volumes
        if volume.external:
            continue

        # Attempting to rebind tmpfs volumes breaks: https://github.com/docker/compose/issues/4751
        if volume.internal in convert_tmpfs_mounts(tmpfs_option).keys():
            continue

        mount = container_mounts.get(volume.internal)

        # New volume, doesn't exist in the old container
        if not mount:
            continue

        # Volume was previously a host volume, now it's a container volume
        if not mount.get('Name'):
            continue

        # Copy existing volume from old container
        volume = volume._replace(external=mount['Name'])
        volumes.append(volume)

    updated_mounts = False
    for mount in mounts_option:
        if mount.type != 'volume':
            continue

        ctnr_mount = container_mounts.get(mount.target)
        if not ctnr_mount.get('Name'):
            continue

        mount.source = ctnr_mount['Name']
        updated_mounts = True

    return volumes, updated_mounts