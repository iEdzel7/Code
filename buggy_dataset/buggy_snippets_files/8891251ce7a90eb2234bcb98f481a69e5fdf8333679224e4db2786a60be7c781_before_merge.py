def delete_cluster(module, redshift):
    """
    Delete a cluster.

    module: Ansible module object
    redshift: authenticated redshift connection object
    """

    identifier = module.params.get('identifier')
    wait = module.params.get('wait')
    wait_timeout = module.params.get('wait_timeout')
    skip_final_cluster_snapshot = module.params.get('skip_final_cluster_snapshot')
    final_cluster_snapshot_identifier = module.params.get('final_cluster_snapshot_identifier')

    try:
        redshift.delete_cluster(
            identifier,
            skip_final_cluster_snapshot,
            final_cluster_snapshot_identifier
        )
    except boto.exception.JSONResponseError as e:
        module.fail_json(msg=str(e))

    if wait:
        try:
            wait_timeout = time.time() + wait_timeout
            resource = redshift.describe_clusters(identifier)['DescribeClustersResponse']['DescribeClustersResult']['Clusters'][0]

            while wait_timeout > time.time() and resource['ClusterStatus'] != 'deleting':
                time.sleep(5)
                if wait_timeout <= time.time():
                    module.fail_json(msg="Timeout waiting for resource %s" % resource.id)

                resource = redshift.describe_clusters(identifier)['DescribeClustersResponse']['DescribeClustersResult']['Clusters'][0]

        except boto.exception.JSONResponseError as e:
            module.fail_json(msg=str(e))

    return(True, {})